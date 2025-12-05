import logging

import torch as th
import torch.nn as nn

from freqtrade.freqai.prediction_models.TorchReinforcementLearner import TorchReinforcementLearner

# ============================================================
# 内存限制: 使用简单 MLP 后，12GB 绰绰有余
# ============================================================
MAX_GPU_MEMORY_GB = 12.0

if th.cuda.is_available():
    total_mem = th.cuda.get_device_properties(0).total_memory
    fraction = (MAX_GPU_MEMORY_GB * 1e9) / total_mem
    fraction = min(fraction, 0.95)  # Safety cap at 95%
    th.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(f"[FlagTraderTorchRL] GPU memory limit: {MAX_GPU_MEMORY_GB}GB")

logger = logging.getLogger(__name__)


class SimpleBackbone(nn.Module):
    """
    Simple MLP Backbone for feature extraction.
    Replaces LLM to drastically reduce memory usage.
    
    Memory usage: ~1 MB (vs ~11 GB for LLM training)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
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
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"SimpleBackbone created: {input_dim} -> {hidden_dim} -> {output_dim}")
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Total parameters: {total_params:,} (~{total_params * 4 / 1e6:.2f} MB in FP32)")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        # Handle both 1D and 2D inputs
        squeeze_output = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        
        # Ensure float type
        if x.dtype not in [th.float32, th.float16]:
            x = x.float()
        
        out = self.net(x)
        
        if squeeze_output:
            out = out.squeeze(0)
        
        return out


class SimpleValueNet(nn.Module):
    """
    Simple MLP for value estimation (Critic network).
    Output shape: [batch] (scalar per sample, required by TorchRL GAE)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        logger.info(f"SimpleValueNet created: {input_dim} -> {hidden_dim} -> 1")
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        if x.dtype not in [th.float32, th.float16]:
            x = x.float()
        
        # Handle 1D input [features] -> [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
            out = self.net(x)
            return out.squeeze()  # [1, 1] -> scalar
        
        # 2D input [batch, features] -> [batch] (squeeze last dim)
        out = self.net(x)  # [batch, 1]
        return out.squeeze(-1)  # [batch]


class FlagTraderTorchRL(TorchReinforcementLearner):
    """
    TorchRL implementation of FlagTrader using simple MLP networks.
    
    This version uses lightweight MLP instead of LLM to avoid memory issues.
    Memory usage: ~10 MB total (vs ~11 GB for LLM)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.backbone = None
        
    def _get_backbone(self, input_dim: int) -> SimpleBackbone:
        """Get or create the shared backbone network."""
        if self.backbone is None:
            # Get config options
            hidden_dim = self.freqai_info.get("rl_config", {}).get("hidden_dim", 256)
            output_dim = self.freqai_info.get("rl_config", {}).get("embedding_dim", 128)
            
            self.backbone = SimpleBackbone(input_dim, hidden_dim, output_dim)
            
            if th.cuda.is_available():
                self.backbone = self.backbone.cuda()
                logger.info(f"GPU memory after backbone: {th.cuda.memory_allocated() / 1e6:.2f} MB")
        
        return self.backbone

    def _build_net(self, input_dim: int, output_dim: int) -> nn.Module:
        """
        Actor Network: Backbone -> Policy Head
        """
        backbone = self._get_backbone(input_dim)
        
        policy_head = nn.Sequential(
            nn.Linear(backbone.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Logits for action distribution
        )
        
        return nn.Sequential(backbone, policy_head)
        
    def _build_value_net(self, input_dim: int) -> nn.Module:
        """
        Critic Network: Simple MLP for value estimation.
        """
        hidden_dim = self.freqai_info.get("rl_config", {}).get("hidden_dim", 256)
        return SimpleValueNet(input_dim, hidden_dim)
