import copy
import gc
import logging
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import Categorical as CategoricalSpec
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import Compose, GymWrapper, StepCounter, TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Positions


# ============================================================
# 内存限制: 使用简单 MLP 后, 12GB 绰绰有余
# ============================================================
MAX_GPU_MEMORY_GB = 12.0

if th.cuda.is_available():
    total_mem = th.cuda.get_device_properties(0).total_memory
    fraction = (MAX_GPU_MEMORY_GB * 1e9) / total_mem
    fraction = min(fraction, 0.95)  # Safety cap at 95%
    th.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[FlagTraderTorchRL] GPU memory limit: {MAX_GPU_MEMORY_GB}GB")

logger = logging.getLogger(__name__)


class FlattenObsGymWrapper(gym.Wrapper):
    """
    Wrapper that:
    1. Converts Pandas DataFrame/Series observations to numpy arrays
    2. Flattens 2D observations (window_size, features) to 1D (window_size * features)

    This is necessary because TorchRL's GymWrapper doesn't handle 2D observations correctly.
    """

    def __init__(self, env):
        super().__init__(env)
        # Update observation space to be 1D (flattened)
        orig_shape = env.observation_space.shape
        if len(orig_shape) == 2:
            # (window_size, features) -> (window_size * features,)
            flat_size = orig_shape[0] * orig_shape[1]
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(flat_size,),
                dtype=np.float32,
            )
            self._flatten = True
            logger.info(f"FlattenObsGymWrapper: {orig_shape} -> ({flat_size},)")
        else:
            self._flatten = False

    def _process_obs(self, obs):
        """Convert observation to flattened numpy array."""
        if isinstance(obs, (pd.DataFrame, pd.Series)):
            obs = obs.values.astype(np.float32)
        if self._flatten and obs.ndim > 1:
            obs = obs.flatten()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        # TorchRL passes tensor actions - convert to int here at the wrapper level
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info


class MLPBackbone(nn.Module):
    """MLP Backbone for flattened window observations."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self._init_weights()

        logger.info(f"MLPBackbone: {input_dim} -> {hidden_dim} -> {output_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,} (~{total_params * 4 / 1e6:.2f} MB)")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dtype not in [th.float32, th.float16]:
            x = x.float()
        return self.net(x)


class PolicyHead(nn.Module):
    """Policy head that outputs action logits."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


class ValueHead(nn.Module):
    """
    Value head for critic network.
    Output: [batch, 1] shape for TorchRL PPO compatibility.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Keep [batch, 1] shape - TorchRL PPO expects this
        return self.net(x)


class ActorNetwork(nn.Module):
    """Actor network: Backbone -> Policy Head"""

    def __init__(self, backbone: MLPBackbone, output_dim: int):
        super().__init__()
        self.backbone = backbone
        self.policy_head = PolicyHead(backbone.output_dim, output_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        features = self.backbone(x)
        return self.policy_head(features)


class CriticNetwork(nn.Module):
    """
    Critic network: Backbone -> Value Head (shares backbone with Actor)
    Output: [batch, 1] shape for TorchRL PPO compatibility.
    """

    def __init__(self, backbone: MLPBackbone):
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(backbone.output_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        features = self.backbone(x)
        # Keep [batch, 1] shape - TorchRL PPO expects this
        return self.value_head(features)


class FlagTraderTorchRL(ReinforcementLearner):
    """
    TorchRL implementation of FlagTrader.

    This class overrides fit() completely to avoid depending on
    TorchReinforcementLearner which may not be updated in Docker container.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = None
        self.model = None
        self.window_size = self.rl_config.get("window_size", 10)

    class MyRLEnv(ReinforcementLearner.MyRLEnv):
        """
        Custom Environment for FlagTraderTorchRL with PnL-driven rewards.
        Reward shaping designed for better learning signals.
        """

        # Reward constants
        INVALID_ACTION_PENALTY = -2.0
        EXIT_PROFIT_BASE = 10.0
        EXIT_PROFIT_MULTIPLIER = 100
        EXIT_LOSS_BASE = -1.0
        EXIT_LOSS_MULTIPLIER = 50
        ENTRY_REWARD = 1.0
        HOLD_NEUTRAL_PENALTY = -0.5
        HOLD_PROFIT_BASE = 0.5
        HOLD_PROFIT_MULTIPLIER = 20
        HOLD_LOSS_BASE = -0.1
        HOLD_LOSS_MULTIPLIER = 10

        def step(self, action):
            """Override step to ensure action is converted to int for TorchRL compatibility."""
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            return super().step(action)

        def _get_exit_reward(self, current_pnl: float) -> float:
            """Calculate reward for exit actions based on PnL."""
            if current_pnl > 0:
                return self.EXIT_PROFIT_BASE + current_pnl * self.EXIT_PROFIT_MULTIPLIER
            return self.EXIT_LOSS_BASE + current_pnl * self.EXIT_LOSS_MULTIPLIER

        def _get_hold_reward(self, current_pnl: float) -> float:
            """Calculate reward for holding a position."""
            if current_pnl > 0:
                return self.HOLD_PROFIT_BASE + current_pnl * self.HOLD_PROFIT_MULTIPLIER
            return self.HOLD_LOSS_BASE + current_pnl * self.HOLD_LOSS_MULTIPLIER

        def calculate_reward(self, action: int) -> float:
            """Reward function with balanced positive/negative signals."""
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)

            if not self._is_valid(action):
                return self.INVALID_ACTION_PENALTY

            current_pnl = self.get_unrealized_profit()

            # EXIT Actions
            if action == Actions.Long_exit.value:
                if self._position == Positions.Long:
                    return self._get_exit_reward(current_pnl)
                return self.INVALID_ACTION_PENALTY

            if action == Actions.Short_exit.value:
                if self._position == Positions.Short:
                    return self._get_exit_reward(current_pnl)
                return self.INVALID_ACTION_PENALTY

            # ENTRY Actions
            if action == Actions.Long_enter.value:
                if self._position == Positions.Neutral:
                    return self.ENTRY_REWARD
                return self.INVALID_ACTION_PENALTY

            if action == Actions.Short_enter.value:
                if self._position == Positions.Neutral:
                    return self.ENTRY_REWARD
                return self.INVALID_ACTION_PENALTY

            # NEUTRAL / HOLD
            if self._position == Positions.Neutral:
                return self.HOLD_NEUTRAL_PENALTY
            return self._get_hold_reward(current_pnl)

    def _cleanup_memory(self) -> None:
        """Clean up previous model and free GPU memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "backbone") and self.backbone is not None:
            del self.backbone
            self.backbone = None

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory at fit() start: {th.cuda.memory_allocated() / 1e9:.2f} GB")

    def _build_networks(
        self, input_dim: int, output_dim: int, device: th.device
    ) -> tuple[ProbabilisticActor, ValueOperator, nn.Module]:
        """Build actor and critic networks."""
        hidden_dim = self.rl_config.get("hidden_dim", 256)
        embedding_dim = self.rl_config.get("embedding_dim", 128)

        self.backbone = MLPBackbone(input_dim, hidden_dim, embedding_dim).to(device)
        actor_net = ActorNetwork(self.backbone, output_dim).to(device)
        critic_net = CriticNetwork(self.backbone).to(device)

        if th.cuda.is_available():
            logger.info(f"GPU memory after networks: {th.cuda.memory_allocated() / 1e6:.2f} MB")

        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["logits"]
        )

        action_spec = CategoricalSpec(n=output_dim, device=device)

        actor = ProbabilisticActor(
            module=actor_module, spec=action_spec, in_keys=["logits"],
            distribution_class=Categorical, return_log_prob=True,
        ).to(device)

        value_module = ValueOperator(
            module=critic_net, in_keys=["observation"],
        ).to(device)

        return actor, value_module, critic_net

    def _create_env_maker(self, train_df, prices_train, env_info, device: th.device):
        """Create environment factory function for TorchRL collector."""
        # Capture references for closure
        MyRLEnv = self.MyRLEnv
        max_steps = len(train_df)

        def env_maker():
            gym_env = MyRLEnv(df=train_df, prices=prices_train, **env_info)
            gym_env = FlattenObsGymWrapper(gym_env)
            # Use categorical_action_encoding=True to match our Categorical actor output
            torchrl_env = GymWrapper(
                gym_env,
                device=device,
                categorical_action_encoding=True,
            )
            return TransformedEnv(
                torchrl_env,
                Compose(StepCounter(max_steps=max_steps))
            )
        return env_maker

    def fit(self, data_dictionary, dk, **kwargs):
        """Train the agent using TorchRL PPO with SyncDataCollector."""
        self._cleanup_memory()

        # Prepare data
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        prices_train, prices_test = self.build_ohlc_price_dataframes(
            dk.data_dictionary, dk.pair, dk
        )

        self.df_raw = copy.deepcopy(train_df)
        self.set_train_and_eval_environments(data_dictionary, prices_train, prices_test, dk)

        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Get dimensions from dummy env
        env_info = self.pack_env_dict(dk.pair)
        dummy_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_info)
        dummy_env = FlattenObsGymWrapper(dummy_env)
        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.action_space.n
        dummy_env.close()

        logger.info(f"Observation dim: {input_dim}, Action dim: {output_dim}")

        # Build networks
        actor, value_module, critic_net = self._build_networks(input_dim, output_dim, device)

        # Setup training components
        frames_per_batch = self.rl_config.get("train_batch_size", 2048)
        entropy_coef = self.rl_config.get("entropy_coef", 0.05)
        ppo_epochs = self.rl_config.get("ppo_epochs", 10)
        mini_batch_size = self.rl_config.get("mini_batch_size", 64)
        gamma = self.rl_config.get("gamma", 0.99)
        gae_lambda = self.rl_config.get("gae_lambda", 0.95)

        # Create environment maker for collector
        env_maker = self._create_env_maker(train_df, prices_train, env_info, device)

        # Create SyncDataCollector - handles rollout collection automatically
        collector = SyncDataCollector(
            env_maker,
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_timesteps,
            split_trajs=False,
            device=device,
        )

        # Create GAE module - handles advantage estimation automatically
        advantage_module = GAE(
            gamma=gamma,
            lmbda=gae_lambda,
            value_network=value_module,
            average_gae=True,
            vectorized=False,  # Set to False for stability with small batches
        )

        # Create PPO loss module
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_module,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coeff=entropy_coef,
            critic_coeff=1.0,
            loss_critic_type="l2",
            normalize_advantage=True,  # 标准化 advantage 提高训练稳定性
        )
        loss_module.set_keys(advantage="advantage", value_target="value_target")

        # Create optimizer with learning rate scheduler
        learning_rate = self.rl_config.get("learning_rate", 3e-4)
        optimizer = th.optim.Adam(loss_module.parameters(), lr=learning_rate)
        total_batches = total_timesteps // frames_per_batch
        scheduler = th.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_batches,
        )

        # Create replay buffer for PPO updates (on-policy: 无放回采样)
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=mini_batch_size,
            sampler=SamplerWithoutReplacement(),  # PPO on-policy 要求每个样本每 epoch 恰好使用一次
        )

        # Training loop using SyncDataCollector
        logger.info("Starting TorchRL SyncDataCollector training...")
        start_time = time.time()

        for batch_idx, tensordict_data in enumerate(collector):
            # Compute advantages using GAE module
            with th.no_grad():
                advantage_module(tensordict_data)

            # Flatten and add to replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # PPO update epochs
            for _ in range(ppo_epochs):
                for _ in range(frames_per_batch // mini_batch_size):
                    subdata = replay_buffer.sample()
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    optimizer.zero_grad()
                    loss_value.backward()
                    th.nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                    optimizer.step()

            # 更新学习率调度器
            scheduler.step()

            # 清空 replay buffer (on-policy 要求每批数据只使用一次)
            replay_buffer.empty()

            # Logging
            if batch_idx % 5 == 0:
                reward_t = tensordict_data["next", "reward"].reshape(-1)
                action_t = tensordict_data["action"].reshape(-1)
                current_lr = scheduler.get_last_lr()[0]
                self._log_training_progress(batch_idx, reward_t, action_t, current_lr)

        collector.shutdown()
        logger.info(f"Training finished in {time.time() - start_time:.2f}s")

        self.model = actor

        # Cleanup
        del collector, loss_module, advantage_module, optimizer, scheduler, replay_buffer
        del value_module, critic_net
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {th.cuda.memory_allocated() / 1e9:.2f} GB")

        return actor

    def _log_training_progress(
        self,
        batch_idx: int,
        reward_t: th.Tensor,
        action_t: th.Tensor,
        current_lr: float | None = None,
    ) -> None:
        """Log training progress."""
        avg_reward = reward_t.mean().item()
        action_counts = [(action_t == a).sum().item() for a in range(5)]
        action_names = ["N", "LE", "LX", "SE", "SX"]
        action_dist = ", ".join(
            f"{n}:{c}" for n, c in zip(action_names, action_counts, strict=True)
        )
        r_pos = (reward_t > 0).sum().item()
        r_neg = (reward_t < 0).sum().item()
        lr_str = f", LR: {current_lr:.2e}" if current_lr is not None else ""
        logger.info(
            f"Batch {batch_idx}, Reward: {avg_reward:.4f} "
            f"(+:{r_pos}, -:{r_neg}), Actions: [{action_dist}]{lr_str}"
        )

    def predict(self, unfiltered_df, dk, **kwargs):
        """
        Run inference using the trained TorchRL model.
        Uses parent class method for proper data handling, only overrides rl_model_predict.
        """
        # Use parent class predict which handles data pipeline correctly
        dk.find_features(unfiltered_df)
        filtered_dataframe, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )

        dk.data_dictionary["prediction_features"] = self.drop_ohlc_from_df(
            filtered_dataframe, dk
        )
        dk.data_dictionary["prediction_features"], _, _ = dk.feature_pipeline.transform(
            dk.data_dictionary["prediction_features"], outlier_check=True
        )

        # Call our custom rl_model_predict
        pred_df = self.rl_model_predict(
            dk.data_dictionary["prediction_features"], dk, self.model
        )
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def rl_model_predict(self, dataframe, dk, model):
        """
        Custom RL model prediction using TorchRL.
        This method is called by predict() for each prediction window.
        """
        if model is None:
            logger.error("Model is None. Returning neutral predictions.")
            return pd.DataFrame(
                np.zeros(len(dataframe)),
                columns=dk.label_list,
                index=dataframe.index,
            )

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Batch processing
        batch_size = self.rl_config.get("prediction_batch_size", 256)
        raw_data = dataframe.values
        all_actions = []

        with th.no_grad():
            for i in range(0, len(raw_data), batch_size):
                chunk = raw_data[i : i + batch_size]
                # Flatten if needed
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)

                obs_data = th.tensor(chunk, dtype=th.float32).to(device)
                input_td = TensorDict({"observation": obs_data}, batch_size=[len(obs_data)])
                output_td = model(input_td)

                logits = output_td["logits"]
                actions = logits.argmax(dim=-1).cpu().numpy()
                all_actions.append(actions)

        actions = np.concatenate(all_actions)

        # Debug: 打印 action 分布
        unique, counts = np.unique(actions, return_counts=True)
        action_dist = dict(zip(unique, counts, strict=False))
        logger.info(f"Predict action distribution: {action_dist}")
        logger.info(f"Total predictions: {len(actions)}, do_predict sum: {dk.do_predict.sum()}")

        # Create prediction dataframe with correct index
        pred_df = pd.DataFrame(
            actions,
            columns=dk.label_list,
            index=dataframe.index,
        )

        return pred_df
