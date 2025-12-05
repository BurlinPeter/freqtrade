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
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import Compose, GymWrapper, StepCounter, TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner

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

    def fit(self, data_dictionary, dk, **kwargs):
        """Train the agent using TorchRL PPO."""
        # Memory cleanup
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

        # 1. Prepare data
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        prices_train, prices_test = self.build_ohlc_price_dataframes(
            dk.data_dictionary, dk.pair, dk
        )

        self.df_raw = copy.deepcopy(train_df)
        self.set_train_and_eval_environments(
            data_dictionary, prices_train, prices_test, dk
        )

        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # 2. Create env_maker with our custom wrapper
        def env_maker():
            env_info = self.pack_env_dict(dk.pair)
            gym_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_info)
            # Use our custom wrapper that flattens observations
            gym_env = FlattenObsGymWrapper(gym_env)
            return TransformedEnv(
                GymWrapper(gym_env, device=device),
                Compose(StepCounter(max_steps=len(train_df))),
            )

        # 3. Get observation and action dimensions from dummy env
        dummy_env = env_maker()
        action_spec = dummy_env.action_spec
        output_dim = action_spec.space.n
        # Get flattened observation dimension
        obs_spec = dummy_env.observation_spec
        obs_shape = obs_spec["observation"].shape
        input_dim = obs_shape[-1]  # Flattened dimension
        dummy_env.close()

        logger.info(f"Observation dim: {input_dim}, Action dim: {output_dim}")

        # 4. Build networks
        hidden_dim = self.rl_config.get("hidden_dim", 256)
        embedding_dim = self.rl_config.get("embedding_dim", 128)

        self.backbone = MLPBackbone(input_dim, hidden_dim, embedding_dim).to(device)
        actor_net = ActorNetwork(self.backbone, output_dim).to(device)
        critic_net = CriticNetwork(self.backbone).to(device)

        if th.cuda.is_available():
            logger.info(f"GPU memory after networks: {th.cuda.memory_allocated() / 1e6:.2f} MB")

        # 5. Create TorchRL modules
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["logits"]
        )

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        ).to(device)

        value_module = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
        ).to(device)

        # 6. Create collector and loss
        frames_per_batch = self.rl_config.get("train_batch_size", 2048)

        collector = SyncDataCollector(
            env_maker,
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_timesteps,
            split_trajs=False,
            device=device,
        )

        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_module,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coef=0.001,
            critic_coef=1.0,
            loss_critic_type="l2",
        )
        loss_module.set_keys(advantage="advantage", value_target="value_target")

        advantage_module = GAE(
            gamma=0.99,
            lmbda=0.95,
            value_network=value_module,
            average_gae=True,
            vectorized=False,
        )

        optimizer = th.optim.Adam(loss_module.parameters(), lr=3e-4)

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=self.rl_config.get("mini_batch_size", 64),
        )

        # 7. Training loop
        logger.info("Starting TorchRL training...")
        start_time = time.time()

        for i, tensordict_data in enumerate(collector):
            with th.no_grad():
                advantage_module(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            ppo_epochs = self.rl_config.get("ppo_epochs", 10)
            mini_batch_size = self.rl_config.get("mini_batch_size", 64)

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
                    optimizer.step()

            if i % 10 == 0:
                avg_reward = tensordict_data["next", "reward"].mean().item()
                logger.info(f"Batch {i}, Avg Reward: {avg_reward:.4f}")

        logger.info(f"Training finished in {time.time() - start_time:.2f}s")

        self.model = actor

        # Cleanup
        del collector, loss_module, advantage_module, optimizer, replay_buffer
        del value_module, critic_net

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {th.cuda.memory_allocated() / 1e9:.2f} GB")

        return actor

    def predict(self, unfiltered_df, dk, **kwargs):
        """Run inference using the trained TorchRL model."""
        if self.model is None:
            logger.error("Model is None. Returning neutral predictions.")
            return (
                pd.DataFrame(
                    np.zeros((len(unfiltered_df),)),
                    columns=[self.rl_config.get("target_col", "trend")],
                    index=unfiltered_df.index,
                ),
                np.zeros(len(unfiltered_df), dtype=int),
            )

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

        model = self.model
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Get flattened input dimension from model
        input_dim = self.backbone.input_dim if self.backbone else None

        # Batch processing
        batch_size = self.rl_config.get("prediction_batch_size", 256)
        raw_data = dk.data_dictionary["prediction_features"].values
        all_actions = []

        with th.no_grad():
            for i in range(0, len(raw_data), batch_size):
                chunk = raw_data[i : i + batch_size]
                # Flatten if needed (handle window data)
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)
                elif chunk.ndim == 2 and input_dim and chunk.shape[1] != input_dim:
                    # Data might already be single-row features, not windows
                    pass

                obs_data = th.tensor(chunk, dtype=th.float32).to(device)
                input_td = TensorDict({"observation": obs_data}, batch_size=[len(obs_data)])
                output_td = model(input_td)

                logits = output_td["logits"]
                actions = logits.argmax(dim=-1).cpu().numpy()
                all_actions.append(actions)

        actions = np.concatenate(all_actions)

        pred_df = pd.DataFrame(
            actions,
            columns=[self.rl_config.get("target_col", "trend")],
            index=filtered_dataframe.index,
        )

        return (pred_df, dk.do_predict)
