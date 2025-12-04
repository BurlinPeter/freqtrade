from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Positions


logger = logging.getLogger(__name__)

# Global cache for LLM to prevent memory explosion
_CACHED_LLM = None
_CACHED_TOKENIZER = None
_CACHED_MODEL_PATH = None

# Try importing transformers and peft
try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers or PEFT not installed. "
        "FlagTraderModel requires 'transformers', 'peft', 'accelerate', and 'bitsandbytes'."
    )


class LLMFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that uses an LLM to process observations.
    Observations (numerical) are converted to a text prompt.
    The LLM is used as a shared backbone for Actor and Critic.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 768,
                 feature_names: list[str] | None = None, model_path: str | None = None,
                 use_peft: bool = True):
        # The features_dim should match the LLM's hidden size.
        # We will update it after loading the model.
        super().__init__(observation_space, features_dim)

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install transformers, peft, accelerate, and bitsandbytes.")

        self.feature_names = feature_names
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # Use global cache
        global _CACHED_LLM, _CACHED_TOKENIZER, _CACHED_MODEL_PATH

        # Check if we can reuse the cached model
        if _CACHED_LLM is not None and _CACHED_MODEL_PATH == model_path:
            logger.info(f"Reusing cached LLM from {model_path}...")
            self.tokenizer = _CACHED_TOKENIZER
            self.llm = _CACHED_LLM
        else:
            logger.info(f"Loading LLM from {model_path}...")

            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load Model
            # optimizing for memory if possible
            bnb_config = None
            if th.cuda.is_available():
                try:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=th.float16,
                    )
                except Exception as e:
                    logger.warning(f"Could not configure BitsAndBytes: {e}")

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config if bnb_config else None,
                torch_dtype=th.float16 if th.cuda.is_available() else th.float32,
                device_map="auto" if th.cuda.is_available() else None,
                local_files_only=True
            )

            # Enable PEFT (LoRA)
            if use_peft:
                logger.info("Applying PEFT (LoRA) to LLM...")
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,  # Or CAUSAL_LM, but we use hidden states
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    init_lora_weights="gaussian"  # Use standard gaussian init instead of failing orthogonal
                )
                # We need to make sure we can get hidden states.
                # TaskType.FEATURE_EXTRACTION might be safer or just generic LoRA.
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.print_trainable_parameters()
            
            # Update Cache
            _CACHED_LLM = self.llm
            _CACHED_TOKENIZER = self.tokenizer
            _CACHED_MODEL_PATH = model_path

        # Update features_dim to match LLM hidden size
        if hasattr(self.llm.config, "hidden_size"):
            self._features_dim = self.llm.config.hidden_size
        else:
            # Fallback for some models
            self._features_dim = 768

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Convert observations to prompts, pass through LLM, return last hidden state.
        """
        prompts = self._observations_to_prompts(observations)

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass
        # We want the last hidden state
        # Ensure mixed precision compatibility (if model is float16, we don't need to cast, but output might need casting)
        outputs = self.llm(**inputs, output_hidden_states=True)

        # Get the last hidden state of the last token
        # shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]

        # Use the representation of the last token
        if self.tokenizer.padding_side == "left":
            embedding = last_hidden_state[:, -1, :]
        else:
            # Right padding: find the last non-pad token
            attention_mask = inputs.attention_mask
            last_token_indices = attention_mask.sum(dim=1) - 1
            embedding = last_hidden_state[
                th.arange(last_hidden_state.shape[0], device=self.device),
                last_token_indices
            ]
        
        # CRITICAL FIX: Cast embedding to float32 before returning
        # PPO's policy network expects float32, but LLM (loaded in 4bit/8bit/half) returns float16.
        # This mismatch causes "RuntimeError: mat1 and mat2 must have the same dtype"
        return embedding.to(dtype=th.float32)

    def _observations_to_prompts(self, observations: th.Tensor) -> list[str]:
        """
        Convert batch of observations to text prompts.
        """
        # Move to CPU for string manipulation
        obs_np = observations.detach().cpu().numpy()
        prompts = []

        # We limit the number of features in the prompt to avoid huge context
        # Or use all if manageable.

        for i in range(len(obs_np)):
            obs_row = obs_np[i]

            # Construct a text description
            # "Feature1: 0.12, Feature2: 1.5, ..."
            # Map indices to names if available

            features_text = []
            if self.feature_names:
                for j, val in enumerate(obs_row):
                    if j < len(self.feature_names):
                        name = self.feature_names[j]
                        
                        # Handle potential numpy/tensor types safely
                        val_float = 0.0
                        try:
                            if hasattr(val, 'item'):
                                val_float = float(val.item())
                            else:
                                val_float = float(val)
                        except (TypeError, ValueError):
                            # Fallback for unexpected array shapes, take the first element
                            if hasattr(val, '__getitem__') and len(val) > 0:
                                val_float = float(val[0])
                        
                        features_text.append(f"{name}: {val_float:.4f}")
            else:
                # Similar safe handling for the else block
                for j, val in enumerate(obs_row):
                    val_float = 0.0
                    try:
                        if hasattr(val, 'item'):
                            val_float = float(val.item())
                        else:
                            val_float = float(val)
                    except (TypeError, ValueError):
                        if hasattr(val, '__getitem__') and len(val) > 0:
                            val_float = float(val[0])
                    features_text.append(f"Feature_{j}: {val_float:.4f}")

            state_str = ", ".join(features_text)

            prompt = (
                "Analyze the following financial market state and decide the "
                "best trading action.\n"
                f"Market State: {state_str}\n"
                "Available Actions: Neutral, Long Enter, Long Exit, Short Enter, Short Exit.\n"
                "Decision:"
            )
            prompts.append(prompt)

        return prompts


class LLMActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy using LLMFeatureExtractor.
    """
    def __init__(self, observation_space, action_space, lr_schedule,
                 features_extractor_class=LLMFeatureExtractor,
                 features_extractor_kwargs=None,
                 **kwargs):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            **kwargs
        )
    
    def init_weights(self, module, gain=1.0):
        """
        Override weight initialization to be compatible with 8-bit quantized modules.
        Stable Baselines3 default init uses orthogonal_, which fails on ByteTensor (int8) AND HalfTensor (float16).
        We skip init for ANY layer that isn't standard float32 or has incompatible types.
        """
        # Skip if module has no weight attribute
        if not hasattr(module, "weight"):
            return
            
        # Skip initialization for quantized layers (uint8) or Half precision layers (float16)
        # "geqrf_cuda" not implemented for 'Half' is the error for float16
        if module.weight.dtype in [th.uint8, th.int8, th.float16, th.bfloat16]:
             return
        
        # For other layers (standard float32), fall back to standard SB3 init
        super().init_weights(module, gain)


class FlagTraderModel(ReinforcementLearner):
    """
    Implementation of FLAG-TRADER: Fusion LLM-Agent with Gradient-based Reinforcement Learning.
    Uses a local LLM (SmolLM2-135M) as the policy network.
    """

    class MyRLEnv(ReinforcementLearner.MyRLEnv):
        """
        Custom Environment for FlagTrader to implement PnL-driven rewards.
        """
        def calculate_reward(self, action: int) -> float:
            """
            Reward function based on PnL (Profit and Loss).
            Aligned with FLAG-TRADER: rewards are driven by trading outcomes.
            """
            # 1. Penalty for invalid actions
            if not self._is_valid(action):
                return -0.1

            # 2. Calculate PnL
            # unrealized_pnl is a ratio (e.g., 0.01 for 1% profit)
            current_pnl = self.get_unrealized_profit()

            # Scale factor to make rewards meaningful for the network
            # (PPO likes rewards roughly in [-1, 1] or [-10, 10])
            scale_factor = 100.0

            # 3. Action-specific rewards

            # EXIT Actions: Realize the PnL
            if action in (Actions.Long_exit.value, Actions.Short_exit.value):
                if self._position != Positions.Neutral:
                    # Reward is the realized PnL
                    return float(current_pnl * scale_factor)
                else:
                    # Invalid exit (should have been caught by _is_valid, but safety net)
                    return -0.1

            # ENTRY Actions:
            # Unlike the default FreqAI implementation, we DO NOT reward entering blindly.
            # The reward for entering comes later when we exit with profit.
            # We might apply a tiny penalty to account for transaction costs immediately.
            if action in (Actions.Long_enter.value, Actions.Short_enter.value):
                if self._position == Positions.Neutral:
                    # Apply transaction fee penalty (simulated)
                    fee = self.fee if hasattr(self, "fee") else 0.001
                    return -float(fee * scale_factor)

            # HOLD / NEUTRAL:
            # We can reward "unrealized PnL change" to shape the reward,
            # or keep it sparse (0). Sparse is harder to train but unbiased.
            # Let's use a small time penalty to encourage capital efficiency?
            # Or simply 0 to strictly follow "PnL" logic.
            # FLAG-TRADER implies "trading rewards", usually PnL.

            # If we are in a position, the reward is the change in PnL since last step
            # (Differential Reward). FreqAI's get_unrealized_profit is cumulative for the trade.
            # To implement differential reward correctly, we'd need to track last_pnl.
            # Given BaseEnv limitations, returning 0 here and full PnL on exit is a safe,
            # sparse approach.

            return 0.0

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        """
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        # Path to the LLM
        # User should ideally configure this in config.json, but we defaults to the known path
        default_model_path = Path("user_data/models/llm/SmolLM2-135M-Instruct")
        model_path = self.freqai_info["rl_config"].get(
            "llm_model_path", str(default_model_path)
        )

        if not Path(model_path).exists():
            # Fallback or error
            logger.error(f"LLM model path not found: {model_path}")
            # Proceeding might fail if internet is restricted or model not cached

        policy_kwargs = dict(
            features_extractor_class=LLMFeatureExtractor,
            features_extractor_kwargs=dict(
                feature_names=dk.training_features_list,
                model_path=model_path,
                use_peft=self.freqai_info["rl_config"].get("use_peft", True)
            ),
            # We might want a small MLP on top of LLM embeddings before the heads
            net_arch=dict(pi=[64], vf=[64])
        )

        if self.activate_tensorboard:
            tb_path = Path(dk.full_path / "tensorboard" / dk.pair.split("/")[0])
        else:
            tb_path = None

        # We use PPO as the base algorithm (Algorithm 2 in paper uses PPO)
        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = PPO(
                LLMActorCriticPolicy,
                self.train_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=tb_path,
                verbose=1,
                learning_rate=self.freqai_info["rl_config"].get("learning_rate", 3e-4),
                **self.freqai_info.get("model_training_parameters", {}),
            )
        else:
            logger.info("Continual training activated - starting from previously trained agent.")
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        callbacks = [self.eval_callback, self.tensorboard_callback]

        try:
            model.learn(
                total_timesteps=int(total_timesteps),
                callback=callbacks,
            )
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e

        return model
