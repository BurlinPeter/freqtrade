import logging
from pathlib import Path

import torch as th
import torch.nn as nn

from freqtrade.freqai.prediction_models.TorchReinforcementLearner import TorchReinforcementLearner


logger = logging.getLogger(__name__)

# Global cache for LLM to prevent memory explosion (Copied from FlagTraderModel)
_CACHED_LLM = None
_CACHED_TOKENIZER = None
_CACHED_MODEL_PATH = None

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "Transformers or PEFT not installed. "
        "FlagTraderTorchRL requires 'transformers', 'peft', 'accelerate', and 'bitsandbytes'."
    )

class LLMBackbone(nn.Module):
    """
    LLM Backbone network shared by Actor and Critic.
    Converts numerical observations to text prompts, passes through LLM, returns embedding.
    """
    def __init__(self, input_dim: int, model_path: str, use_peft: bool = True):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Please install transformers, peft, accelerate, and bitsandbytes.")

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        # Use global cache
        global _CACHED_LLM, _CACHED_TOKENIZER, _CACHED_MODEL_PATH

        if _CACHED_LLM is not None and _CACHED_MODEL_PATH == model_path:
            logger.info(f"Reusing cached LLM from {model_path}...")
            self.tokenizer = _CACHED_TOKENIZER
            self.llm = _CACHED_LLM
        else:
            logger.info(f"Loading LLM from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

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

            if use_peft:
                logger.info("Applying PEFT (LoRA) to LLM...")
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    inference_mode=False,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    init_lora_weights="gaussian"
                )
                self.llm = get_peft_model(self.llm, peft_config)
                self.llm.print_trainable_parameters()
            
            _CACHED_LLM = self.llm
            _CACHED_TOKENIZER = self.tokenizer
            _CACHED_MODEL_PATH = model_path

        if hasattr(self.llm.config, "hidden_size"):
            self.output_dim = self.llm.config.hidden_size
        else:
            self.output_dim = 768

    def forward(self, observations: th.Tensor) -> th.Tensor:
        prompts = self._observations_to_prompts(observations)

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Run LLM
        outputs = self.llm(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]

        if self.tokenizer.padding_side == "left":
            embedding = last_hidden_state[:, -1, :]
        else:
            attention_mask = inputs.attention_mask
            last_token_indices = attention_mask.sum(dim=1) - 1
            embedding = last_hidden_state[
                th.arange(last_hidden_state.shape[0], device=self.device),
                last_token_indices
            ]
        
        return embedding.to(dtype=th.float32)

    def _observations_to_prompts(self, observations: th.Tensor) -> list[str]:
        # Simplified prompt generation logic
        obs_np = observations.detach().cpu().numpy()
        prompts = []
        for i in range(len(obs_np)):
            obs_row = obs_np[i]
            features_text = [f"Feat_{j}: {val:.4f}" for j, val in enumerate(obs_row)]
            state_str = ", ".join(features_text)
            prompt = (
                "Analyze market state and decide action.\n"
                f"State: {state_str}\n"
                "Decision:"
            )
            prompts.append(prompt)
        return prompts


class FlagTraderTorchRL(TorchReinforcementLearner):
    """
    TorchRL implementation of FlagTrader (LLM-based RL).
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_backbone = None
        
    def _get_llm_backbone(self, input_dim):
        # Singleton-like access to ensure we share the backbone between actor and critic
        # or just re-create references if cached.
        if self.llm_backbone is None:
             # Get model path from config
            default_model_path = Path("user_data/models/llm/SmolLM2-135M-Instruct")
            model_path = self.freqai_info["rl_config"].get(
                "llm_model_path", str(default_model_path)
            )
            use_peft = self.freqai_info["rl_config"].get("use_peft", True)
            
            self.llm_backbone = LLMBackbone(input_dim, model_path, use_peft)
        return self.llm_backbone

    def _build_net(self, input_dim, output_dim):
        """
        Actor Network: LLM Backbone -> Head
        """
        backbone = self._get_llm_backbone(input_dim)
        
        return nn.Sequential(
            backbone,
            nn.Linear(backbone.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim) # Logits
        )
        
    def _build_value_net(self, input_dim):
        """
        Critic Network: LLM Backbone -> Head
        """
        backbone = self._get_llm_backbone(input_dim)
        
        return nn.Sequential(
            backbone,
            nn.Linear(backbone.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Value
        )
