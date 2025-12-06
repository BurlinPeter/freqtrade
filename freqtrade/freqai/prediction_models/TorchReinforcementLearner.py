import copy
import logging
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch as th
import torch.nn as nn
from pandas import DataFrame
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import Compose, GymWrapper, ObservationTransform, StepCounter, TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner


logger = logging.getLogger(__name__)


class LLMTokenizerTransform(ObservationTransform):
    """
    Transform that converts numerical observations to LLM input_ids using the model's tokenizer.
    This runs on the CPU/Env side to avoid string operations in the GPU/vmap graph.
    """
    def __init__(self, tokenizer, prompt_func, max_length=512):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
        self.tokenizer = tokenizer
        self.prompt_func = prompt_func
        self.max_length = max_length

    def transform_observation_spec(self, observation_spec):
        """Update the observation spec to reflect tokenized output shape."""
        from torchrl.data import Bounded, Composite

        # observation_spec is a Composite containing "observation" key
        # We need to replace the "observation" spec with the tokenized version
        vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 50000
        device = observation_spec.device if hasattr(observation_spec, 'device') else 'cpu'

        new_obs_spec = Bounded(
            low=0,
            high=vocab_size,
            shape=(self.max_length,),
            dtype=th.int64,
            device=device,
        )

        # Return a new Composite with the updated observation spec
        return Composite(
            observation=new_obs_spec,
            device=device,
        )

    def _apply_transform(self, obs: th.Tensor) -> th.Tensor:
        # obs is a Tensor (from PandasToNumpyWrapper), likely on CPU or CUDA
        # Convert to numpy for prompt generation
        try:
            obs_np = obs.detach().cpu().numpy()
        except Exception as e:
            # If conversion fails, return a zero-padded tensor of correct shape
            # to maintain consistent observation dimensions
            logger.warning(f"LLMTokenizerTransform: Failed to convert obs to numpy: {e}")
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            batch_size = obs.shape[0] if obs.dim() > 1 else 1
            return th.full((batch_size, self.max_length), pad_id, dtype=th.int64, device=obs.device)

        # Handle unbatched input (Gym env usually returns single obs)
        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]
            squeeze = True
        else:
            squeeze = False

        prompts = self.prompt_func(obs_np)

        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=False, # Disable tokenizer padding, do it manually
            truncation=True,
            max_length=self.max_length
        )

        input_ids = tokens["input_ids"] # (Batch, VarLen)

        # Aggressive Manual Padding / Truncation by pre-allocating tensor
        batch_size, seq_len = input_ids.shape
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        final_input_ids = th.full(
            (batch_size, self.max_length),
            pad_id,
            dtype=input_ids.dtype,
            device=input_ids.device
        )

        # Copy valid tokens
        valid_len = min(seq_len, self.max_length)
        final_input_ids[:, :valid_len] = input_ids[:, :valid_len]

        if squeeze:
            final_input_ids = final_input_ids.squeeze(0)

        return final_input_ids.to(obs.device)


class PandasToNumpyGymWrapper(gym.Wrapper):
    """
    Wrapper to ensure observations are numpy arrays instead of Pandas DataFrame/Series.
    FreqAI environments typically return Pandas objects, which TorchRL hates.

    Also flattens 2D observations (window_size, features) to 1D (window_size * features)
    because TorchRL's GymWrapper doesn't handle 2D observations correctly.
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
        else:
            self._flatten = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, (pd.DataFrame, pd.Series)):
            obs = obs.values.astype(np.float32)
        if self._flatten and obs.ndim > 1:
            obs = obs.flatten()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, (pd.DataFrame, pd.Series)):
            obs = obs.values.astype(np.float32)
        if self._flatten and obs.ndim > 1:
            obs = obs.flatten()
        return obs, reward, terminated, truncated, info


class TokenizingGymWrapper(gym.Wrapper):
    """
    Gym wrapper that tokenizes observations at the Gym level (before TorchRL).
    This ensures consistent observation shapes for both reset() and step().
    """
    def __init__(self, env, tokenizer, prompt_func, max_length=512):
        super().__init__(env)
        self.tokenizer = tokenizer
        self.prompt_func = prompt_func
        self.max_length = max_length

        # Update observation space to reflect tokenized output
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
        self.observation_space = gym.spaces.Box(
            low=0, high=vocab_size, shape=(max_length,), dtype=np.int64
        )

    def _tokenize_obs(self, obs):
        """Convert observation to tokenized input_ids."""
        # Ensure obs is numpy array
        if isinstance(obs, (pd.DataFrame, pd.Series)):
            obs = obs.values.astype(np.float32)

        # Handle 1D obs (single sample)
        if obs.ndim == 1:
            obs = obs[None, :]
            squeeze = True
        else:
            squeeze = False

        # Generate prompts
        prompts = self.prompt_func(obs)

        # Tokenize
        tokens = self.tokenizer(
            prompts,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.max_length
        )

        input_ids = tokens["input_ids"]

        # Pad to fixed length
        batch_size, seq_len = input_ids.shape
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        final_ids = np.full((batch_size, self.max_length), pad_id, dtype=np.int64)
        valid_len = min(seq_len, self.max_length)
        final_ids[:, :valid_len] = input_ids[:, :valid_len]

        if squeeze:
            final_ids = final_ids.squeeze(0)

        return final_ids

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        tokenized_obs = self._tokenize_obs(obs)
        return tokenized_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        tokenized_obs = self._tokenize_obs(obs)
        return tokenized_obs, reward, terminated, truncated, info


class TorchReinforcementLearner(ReinforcementLearner):
    """
    TorchRL based Reinforcement Learning Model.
    Inherits from ReinforcementLearner to reuse environment setup and data handling,
    but overrides fit() and predict() to use TorchRL instead of Stable-Baselines3.
    """

    def _cleanup_previous_model(self) -> None:
        """Release previous model and LLM backbone to free GPU memory."""
        import gc

        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None

        if hasattr(self, 'backbone') and self.backbone is not None:
            del self.backbone
            self.backbone = None

        if hasattr(self, 'llm_backbone') and self.llm_backbone is not None:
            del self.llm_backbone
            self.llm_backbone = None

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            mem_gb = th.cuda.memory_allocated() / 1e9
            logger.info(f"GPU memory at fit() start (after cleanup): {mem_gb:.2f} GB")

    def _create_env_maker(
        self, train_df: DataFrame, prices_train: DataFrame, dk: FreqaiDataKitchen,
        device: th.device, tokenizer: Any, prompt_func: Any, use_llm_tokenizer: bool
    ):
        """Create environment factory function for TorchRL collector."""
        def env_maker():
            env_info = self.pack_env_dict(dk.pair)
            gym_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_info)
            gym_env = PandasToNumpyGymWrapper(gym_env)

            if use_llm_tokenizer:
                gym_env = TokenizingGymWrapper(gym_env, tokenizer, prompt_func)

            return TransformedEnv(
                GymWrapper(gym_env, device=device),
                Compose(StepCounter(max_steps=len(train_df)))
            )
        return env_maker

    def _build_actor_critic(
        self, input_dim: int, output_dim: int, action_spec: Any, device: th.device
    ) -> tuple[ProbabilisticActor, ValueOperator, nn.Module]:
        """Build actor and critic networks for PPO."""
        net = self._build_net(input_dim, output_dim).to(device)

        actor_module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["logits"]
        )

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        ).to(device)

        value_net = self._build_value_net(input_dim).to(device)
        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        ).to(device)

        return actor, value_module, value_net

    def _run_training_loop(
        self, collector: SyncDataCollector, loss_module: ClipPPOLoss,
        advantage_module: GAE, optimizer: th.optim.Optimizer,
        replay_buffer: ReplayBuffer, frames_per_batch: int, device: th.device
    ) -> None:
        """Execute the PPO training loop."""
        logger.info("Starting TorchRL training...")
        start_time = time.time()

        ppo_epochs = self.rl_config.get("ppo_epochs", 10)
        mini_batch_size = self.rl_config.get("mini_batch_size", 64)

        for i, tensordict_data in enumerate(collector):
            with th.no_grad():
                advantage_module(tensordict_data)

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

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

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """Train the agent using TorchRL PPO."""
        self._cleanup_previous_model()

        # Prepare data and environment
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        prices_train, prices_test = self.build_ohlc_price_dataframes(
            dk.data_dictionary, dk.pair, dk
        )

        _ = dk.make_train_test_datasets(train_df, data_dictionary["train_labels"])
        self.df_raw = copy.deepcopy(train_df)
        self.set_train_and_eval_environments(
            data_dictionary, prices_train, prices_test, dk
        )

        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Pre-fetch tokenizer and prompt function
        input_dim = train_df.shape[1]
        use_llm_tokenizer = hasattr(self, "_get_llm_backbone")
        if use_llm_tokenizer:
            backbone = self._get_llm_backbone(input_dim)
            tokenizer = backbone.tokenizer
            prompt_func = backbone.manual_observations_to_prompts
        else:
            tokenizer = None
            prompt_func = None

        # Create environment factory
        env_maker = self._create_env_maker(
            train_df, prices_train, dk, device, tokenizer, prompt_func, use_llm_tokenizer
        )

        # Get action dimensions from dummy env
        dummy_env = env_maker()
        action_spec = dummy_env.action_spec
        output_dim = action_spec.space.n
        dummy_env.close()

        # Build networks
        actor, value_module, value_net = self._build_actor_critic(
            input_dim, output_dim, action_spec, device
        )

        # Setup training components
        frames_per_batch = self.rl_config.get("train_batch_size", 2048)

        collector = SyncDataCollector(
            env_maker, policy=actor, frames_per_batch=frames_per_batch,
            total_frames=total_timesteps, split_trajs=False, device=device,
        )

        loss_module = ClipPPOLoss(
            actor_network=actor, critic_network=value_module, clip_epsilon=0.2,
            entropy_bonus=True, entropy_coef=0.001, critic_coef=1.0,
            loss_critic_type="l2",
        )
        loss_module.set_keys(advantage="advantage", value_target="value_target")

        advantage_module = GAE(
            gamma=0.99, lmbda=0.95, value_network=value_module,
            average_gae=True, vectorized=False,
        )

        optimizer = th.optim.Adam(loss_module.parameters(), lr=3e-4)

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=self.rl_config.get("mini_batch_size", 64),
        )

        # Run training
        self._run_training_loop(
            collector, loss_module, advantage_module, optimizer,
            replay_buffer, frames_per_batch, device
        )

        self.model = actor

        # Cleanup
        del collector, loss_module, advantage_module, optimizer, replay_buffer
        del value_module, value_net

        import gc
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {th.cuda.memory_allocated() / 1e9:.2f} GB")

        return actor

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        Run inference using the trained TorchRL model.
        """
        # Check if model is loaded
        if self.model is None:
            logger.warning(
                "Model not initialized in predict(). Attempting to load or rebuilding..."
            )
            # Try to infer dims from data to rebuild net if needed, though load() is preferred.
            # If we are here, it means load() failed or wasn't called, and fit() wasn't called.
            # FreqAI flow: fit() -> predict() OR load() -> predict()
            # If load failed, we can't do much.
            # But we can try to be robust if it's just uninitialized architecture.

            # NOTE: We cannot easily rebuild without knowing if it was loaded.
            # If it is None, we return zeros/neutral to avoid crash, or raise error.
            logger.error("Self.model is None. Returning neutral predictions.")
            return (
                pd.DataFrame(
                    np.zeros((len(unfiltered_df),)),
                    columns=[self.rl_config.get("target_col", "trend")],
                    index=unfiltered_df.index
                ),
                np.zeros(len(unfiltered_df), dtype=int)
            )

        # Re-use parent's data preparation
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

        # Custom inference logic for TorchRL
        model = self.model
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Batch processing to prevent OOM with LLMs or large datasets
        batch_size = self.rl_config.get("prediction_batch_size", 256)
        input_data = dk.data_dictionary["prediction_features"].values
        all_actions = []

        with th.no_grad():
            for i in range(0, len(input_data), batch_size):
                chunk = input_data[i : i + batch_size]
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

    def _build_net(self, input_dim, output_dim):
        """
        Default simple MLP. Override this for LLM or other architectures.
        """
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),  # Logits
        )

    def _build_value_net(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def save(self, path: Path) -> None:
        """
        Save the model state_dict to a file.
        Overrides parent method to handle PyTorch models safely.
        """
        if self.model:
            # We save the state_dict of the actor network
            # If we wanted to save the critic or optimizer, we'd need to bundle them.
            # For inference, only actor is needed.
            save_path = path / "actor_state_dict.pt"
            th.save(self.model.state_dict(), save_path)
            logger.info(f"Saved TorchRL actor model to {save_path}")
        else:
            logger.warning("No model to save.")

    def load(self, path: Path) -> None:
        """
        Load the model state_dict from a file.
        """
        load_path = path / "actor_state_dict.pt"
        if load_path.is_file():
            logger.info(f"Loading TorchRL actor model from {load_path}")
            # We assume self.model has been initialized with the correct architecture
            # by the caller (e.g. during fit or before predict).
            # If not, this will fail.
            if self.model is None:
                 logger.warning("Model structure not initialized. Cannot load weights.")
                 return

            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            state_dict = th.load(load_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
        else:
            logger.warning(f"Could not find model file at {load_path}")
