"""
FLAG-TRADER: LLM-based Reinforcement Learning for Financial Trading

Implementation based on the paper:
"FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading"

Key features:
1. Uses SmolLM2-135M-Instruct as the backbone
2. Partial layer freezing for parameter-efficient fine-tuning
3. PPO training with shared LLM backbone for actor and critic
4. Prompt-based state representation
"""

import copy
import gc
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from pandas import DataFrame
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import Categorical as CategoricalSpec
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import Compose, GymWrapper, StepCounter, TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Positions


logger = logging.getLogger(__name__)

# ============================================================
# GPU Memory Limit
# ============================================================
MAX_GPU_MEMORY_GB = 24.0

if th.cuda.is_available():
    total_mem = th.cuda.get_device_properties(0).total_memory
    fraction = (MAX_GPU_MEMORY_GB * 1e9) / total_mem
    fraction = min(fraction, 0.95)
    th.cuda.set_per_process_memory_fraction(fraction, device=0)
    logger.info(f"[FlagTraderLLM] GPU memory limit: {MAX_GPU_MEMORY_GB}GB")


# ============================================================
# Prompt Builder
# ============================================================
class PromptBuilder:
    """
    将数值市场状态转换为LLM可理解的文本prompt。
    参考FLAG-TRADER论文的prompt设计。
    
    优化: 
    1. Action Space 放在开头，确保不会被截断
    2. 持仓状态信息单独显示，便于LLM理解
    """

    # 系统提示（放在最前面，确保不被截断）
    SYSTEM_PROMPT = """You are a crypto trader. Choose action 0-4:
0=Hold, 1=Long_Enter, 2=Long_Exit, 3=Short_Enter, 4=Short_Exit
Rules: Long_Enter/Short_Enter only when Neutral. Long_Exit only when Long. Short_Exit only when Short."""

    def __init__(self, feature_names: list[str] | None = None, max_features: int = 50):
        """
        Args:
            feature_names: 特征名称列表
            max_features: 最大显示特征数（避免prompt过长）
        """
        self.feature_names = feature_names
        self.max_features = max_features

    def _parse_position_value(self, pos_val: float) -> str:
        """将position数值转换为可读字符串"""
        if pos_val <= 0.25:
            return "Short"
        elif pos_val >= 0.75:
            return "Long"
        else:
            return "Neutral"

    def build_prompt(self, obs: np.ndarray) -> str:
        """
        构建单个observation的prompt。

        Args:
            obs: 1D numpy array of features
        """
        # 分离状态信息和技术指标 (Portfolio Features)
        position_str = "Neutral"
        unrealized_pnl_str = "0.00%"
        duration_str = "0"
        total_profit_str = "0.00%"
        cash_ratio_str = "100%"
        
        feature_parts = []
        state_feature_count = 0  # 统计状态特征数量
        
        if self.feature_names is not None:
            for i, (name, val) in enumerate(zip(self.feature_names, obs, strict=False)):
                clean_name = name.lstrip("%-")
                
                # 提取 Portfolio Features（状态信息）
                if clean_name == "position":
                    position_str = self._parse_position_value(val)
                    state_feature_count += 1
                elif clean_name == "unrealized_pnl":
                    unrealized_pnl_str = f"{val*100:+.2f}%"  # 带符号显示
                    state_feature_count += 1
                elif clean_name == "trade_duration":
                    duration_str = str(int(val))
                    state_feature_count += 1
                elif clean_name == "total_profit":
                    total_profit_str = f"{val*100:+.2f}%"  # 带符号显示
                    state_feature_count += 1
                elif clean_name == "cash_ratio":
                    cash_ratio_str = f"{val*100:.0f}%"
                    state_feature_count += 1
                elif i < self.max_features + state_feature_count:
                    # 普通技术指标 (Market Features)
                    feature_parts.append(f"{clean_name}:{val:.4f}")
            
            feature_str = ", ".join(feature_parts)
            remaining = len(obs) - len(feature_parts) - state_feature_count
            if remaining > 0:
                feature_str += f" (+{remaining} more)"
        else:
            # 没有feature names时使用简单格式
            feature_str = ", ".join(f"{v:.4f}" for v in obs[:10]) + "..."

        # 构建prompt：Portfolio状态在前，Market数据在后
        prompt = f"""{self.SYSTEM_PROMPT}

Portfolio: Pos={position_str}, Cash={cash_ratio_str}, UnrealizedPnL={unrealized_pnl_str}, TotalPnL={total_profit_str}, Duration={duration_str}
Market: {feature_str}

Action?"""

        return prompt

    def batch_build_prompts(self, obs_batch: np.ndarray) -> list[str]:
        """批量构建prompts"""
        # 确保 obs_batch 是2D数组 [batch, features]
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        return [self.build_prompt(obs) for obs in obs_batch]


# ============================================================
# LLM Backbone with Partial Freezing
# ============================================================
class LLMBackbone(nn.Module):
    """
    LLM Backbone for FLAG-TRADER.
    使用SmolLM2-135M-Instruct作为特征提取器。
    实现部分参数冻结策略（论文核心技术）。
    """

    def __init__(
        self,
        model_path: str,
        freeze_layers: int = 25,  # 冻结前25层，训练最后5层 (SmolLM2有30层)
        max_length: int = 512,
        dtype: th.dtype = th.bfloat16,  # bfloat16 比 float16 数值更稳定
    ):
        super().__init__()

        self.max_length = max_length
        self.dtype = dtype

        # 加载tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载LLM
        logger.info(f"Loading LLM from {model_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # 应用冻结策略
        self._freeze_layers(freeze_layers)

        # 输出维度 = hidden_size
        self.output_dim = self.llm.config.hidden_size  # 576 for SmolLM2-135M

        logger.info(f"LLMBackbone loaded successfully")
        logger.info(f"Hidden size: {self.output_dim}, Total layers: {len(self.llm.model.layers)}")
        logger.info(f"Frozen layers: {freeze_layers}, Trainable layers: {len(self.llm.model.layers) - freeze_layers}")
        self._log_trainable_params()

    def _freeze_layers(self, freeze_layers: int):
        """
        冻结前N层transformer层。
        
        根据FLAG-TRADER论文:
        - θ_frozen: 冻结的参数（前面的层）
        - θ_train: 可训练的参数（后面的层）
        """
        total_layers = len(self.llm.model.layers)

        # 1. 冻结embeddings (通常不需要微调)
        for param in self.llm.model.embed_tokens.parameters():
            param.requires_grad = False

        # 2. 冻结前freeze_layers层，训练后面的层
        for i, layer in enumerate(self.llm.model.layers):
            should_freeze = i < freeze_layers
            for param in layer.parameters():
                param.requires_grad = not should_freeze

        # 3. 保持最后的LayerNorm可训练
        for param in self.llm.model.norm.parameters():
            param.requires_grad = True

        # 4. 冻结lm_head (我们不需要做语言建模)
        if hasattr(self.llm, 'lm_head'):
            for param in self.llm.lm_head.parameters():
                param.requires_grad = False

        logger.info(f"Froze first {freeze_layers}/{total_layers} transformer layers")

    def _log_trainable_params(self):
        """统计可训练参数"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        logger.info(f"Parameter Statistics:")
        logger.info(f"  Total:     {total_params:>12,} params")
        logger.info(f"  Trainable: {trainable_params:>12,} params ({100*trainable_params/total_params:.2f}%)")
        logger.info(f"  Frozen:    {frozen_params:>12,} params ({100*frozen_params/total_params:.2f}%)")

    def forward(self, input_ids: th.Tensor, attention_mask: th.Tensor | None = None) -> th.Tensor:
        """
        前向传播，返回最后一个token的hidden state。

        Args:
            input_ids: [batch, seq_len] token IDs
            attention_mask: [batch, seq_len] attention mask

        Returns:
            hidden_states: [batch, hidden_size] 最后一个有效token的hidden state
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 获取LLM的hidden states
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 获取最后一层的hidden states
        last_hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

        # 获取每个样本最后一个有效token的hidden state
        batch_size = input_ids.shape[0]
        seq_lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效token的index
        seq_lengths = seq_lengths.clamp(min=0)  # 确保不为负

        # 使用advanced indexing获取最后一个token的hidden state
        last_token_hidden = last_hidden[
            th.arange(batch_size, device=input_ids.device),
            seq_lengths
        ]  # [batch, hidden_size]

        # 转回float32并进行数值稳定性处理
        result = last_token_hidden.float()
        
        # 检测并处理 NaN/Inf（在源头处理，而不是在下游）
        if th.isnan(result).any() or th.isinf(result).any():
            # 使用更保守的替换策略：用该batch的均值填充异常值
            mask = th.isnan(result) | th.isinf(result)
            valid_mean = result[~mask].mean() if (~mask).any() else 0.0
            result = th.where(mask, th.full_like(result, valid_mean), result)
        
        return result

    def tokenize(self, texts: list[str], device: th.device) -> tuple[th.Tensor, th.Tensor]:
        """将文本tokenize为input_ids和attention_mask"""
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # 打印实际 token 数量
        actual_length = tokens["input_ids"].shape[1]
        if actual_length >= self.max_length:
            logger.warning(
                f"⚠️ Prompt truncated! Token count: {actual_length} >= max_length: {self.max_length}"
            )
        elif not hasattr(self, "_token_count_logged"):
            # 只在第一次打印，避免刷屏
            logger.info(f"Prompt token count: {actual_length}/{self.max_length}")
            self._token_count_logged = True
        
        return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)


# ============================================================
# Policy and Value Networks
# ============================================================
class LLMPolicyNetwork(nn.Module):
    """
    策略网络: LLM Backbone -> Policy Head
    将数值observation转换为prompt，通过LLM提取特征，输出action logits。
    """

    def __init__(
        self,
        llm_backbone: LLMBackbone,
        num_actions: int = 5,
        prompt_builder: PromptBuilder | None = None,
    ):
        super().__init__()
        self.llm_backbone = llm_backbone
        self.prompt_builder = prompt_builder or PromptBuilder()

        # Policy Head: hidden_size -> num_actions
        # 论文设计: 较小的head，依赖LLM的表示能力
        self.policy_head = nn.Sequential(
            nn.Linear(llm_backbone.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_actions),
        )

        self._init_head_weights()
        
        # 保存action数量用于初始化均匀分布
        self.num_actions = num_actions

    def _init_head_weights(self):
        """
        初始化head权重。
        
        关键：最后一层使用更小的初始化范围，确保初始logits接近均匀分布，
        避免policy collapse到单一action。
        """
        modules = list(self.policy_head.modules())
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                # 最后一层（输出层）使用更小的初始化
                if i == len(modules) - 1 or m.out_features == getattr(self, 'num_actions', 5):
                    # 小权重 + 零bias → 初始logits接近0 → softmax接近均匀分布
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _materialize_observation(self, observation: th.Tensor) -> np.ndarray:
        """
        将 observation 张量转换为 numpy 数组。
        
        处理 TorchRL vmap 创建的无存储空间的批量张量问题。
        vmap 张量没有实际的 storage，直接调用 .numpy() 会失败。
        """
        try:
            return observation.detach().clone().cpu().numpy()
        except RuntimeError:
            # vmap 创建的批量张量没有 storage，通过 tolist() 实体化
            return np.array(observation.detach().tolist(), dtype=np.float32)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """
        Args:
            observation: [batch, obs_dim] 数值observation

        Returns:
            logits: [batch, num_actions] action logits
        """
        device = observation.device

        # 确保 observation 是2D [batch, features]
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # 将数值observation转换为prompts (处理vmap张量)
        obs_np = self._materialize_observation(observation)
        prompts = self.prompt_builder.batch_build_prompts(obs_np)

        # Tokenize
        input_ids, attention_mask = self.llm_backbone.tokenize(prompts, device)

        # 通过LLM获取hidden states
        hidden_states = self.llm_backbone(input_ids, attention_mask)

        # 检查 hidden_states 的数值稳定性
        if th.isnan(hidden_states).any() or th.isinf(hidden_states).any():
            logger.warning("NaN/Inf detected in hidden_states, replacing with zeros")
            hidden_states = th.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)

        # 通过Policy Head获取action logits
        logits = self.policy_head(hidden_states)

        # 数值稳定性: 裁剪 logits 防止极端值导致 softmax 溢出
        logits = th.clamp(logits, min=-20.0, max=20.0)
        
        # 最终检查
        if th.isnan(logits).any() or th.isinf(logits).any():
            logger.warning("NaN/Inf detected in logits, replacing with zeros")
            logits = th.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)

        return logits


class LLMValueNetwork(nn.Module):
    """
    价值网络: LLM Backbone -> Value Head
    与Policy网络共享LLM Backbone（参数高效）。
    """

    def __init__(
        self,
        llm_backbone: LLMBackbone,
        prompt_builder: PromptBuilder | None = None,
    ):
        super().__init__()
        self.llm_backbone = llm_backbone
        self.prompt_builder = prompt_builder or PromptBuilder()

        # Value Head: hidden_size -> 1
        self.value_head = nn.Sequential(
            nn.Linear(llm_backbone.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self._init_head_weights()

    def _init_head_weights(self):
        modules = list(self.value_head.modules())
        for i, m in enumerate(modules):
            if isinstance(m, nn.Linear):
                # 最后一层（输出层）使用更小的初始化，value初始接近0
                if i == len(modules) - 1 or m.out_features == 1:
                    nn.init.uniform_(m.weight, -0.01, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _materialize_observation(self, observation: th.Tensor) -> np.ndarray:
        """
        将 observation 张量转换为 numpy 数组。
        
        处理 TorchRL vmap 创建的无存储空间的批量张量问题。
        """
        try:
            return observation.detach().clone().cpu().numpy()
        except RuntimeError:
            return np.array(observation.detach().tolist(), dtype=np.float32)

    def forward(self, observation: th.Tensor) -> th.Tensor:
        """
        Args:
            observation: [batch, obs_dim] 数值observation

        Returns:
            value: [batch, 1] state value
        """
        device = observation.device

        # 确保 observation 是2D [batch, features]
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

        # 将数值observation转换为prompts (处理vmap张量)
        obs_np = self._materialize_observation(observation)
        prompts = self.prompt_builder.batch_build_prompts(obs_np)

        # Tokenize
        input_ids, attention_mask = self.llm_backbone.tokenize(prompts, device)

        # 通过LLM获取hidden states
        hidden_states = self.llm_backbone(input_ids, attention_mask)

        # 检查 hidden_states 的数值稳定性
        if th.isnan(hidden_states).any() or th.isinf(hidden_states).any():
            logger.warning("NaN/Inf detected in value hidden_states, replacing with zeros")
            hidden_states = th.nan_to_num(hidden_states, nan=0.0, posinf=1.0, neginf=-1.0)

        # 通过Value Head获取state value
        value = self.value_head(hidden_states)

        # 数值稳定性: 裁剪 value 防止极端值
        value = th.clamp(value, min=-100.0, max=100.0)
        
        if th.isnan(value).any() or th.isinf(value).any():
            logger.warning("NaN/Inf detected in value, replacing with zeros")
            value = th.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)

        return value


# ============================================================
# Gym Wrapper
# ============================================================
class FlattenObsGymWrapper(gym.Wrapper):
    """
    将2D observation展平为1D。
    TorchRL的GymWrapper不能很好地处理2D observations。
    """

    def __init__(self, env):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        if len(orig_shape) == 2:
            flat_size = orig_shape[0] * orig_shape[1]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
            )
            self._flatten = True
            logger.info(f"FlattenObsGymWrapper: {orig_shape} -> ({flat_size},)")
        else:
            self._flatten = False

    def _process_obs(self, obs):
        if isinstance(obs, (pd.DataFrame, pd.Series)):
            obs = obs.values.astype(np.float32)
        if self._flatten and obs.ndim > 1:
            obs = obs.flatten()
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        # 确保 action 是 Python 标量，避免 NumPy deprecation warning
        if hasattr(action, 'item'):
            action = action.item()
        elif isinstance(action, np.ndarray):
            action = action.flat[0] if action.size > 0 else 0
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._process_obs(obs), reward, terminated, truncated, info


# ============================================================
# Main Model Class
# ============================================================
class FlagTraderLLM(ReinforcementLearner):
    """
    FLAG-TRADER: LLM-based Reinforcement Learning Model.

    使用SmolLM2-135M-Instruct作为backbone，
    实现论文中描述的PPO训练流程。

    Key features:
    1. Partial layer freezing for parameter-efficient fine-tuning
    2. Prompt-based state representation
    3. PPO training with shared LLM backbone
    4. Gradient accumulation for stable training
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_backbone = None
        self.model = None
        self.prompt_builder = None
        self.window_size = self.rl_config.get("window_size", 10)

        # 从config获取LLM路径
        self.llm_model_path = self.rl_config.get(
            "llm_model_path",
            "user_data/models/llm/SmolLM2-135M-Instruct"
        )

    class MyRLEnv(ReinforcementLearner.MyRLEnv):
        """
        自定义RL环境，使用PnL驱动的reward设计。
        强制添加持仓信息到observation中，供LLM理解当前状态。
        """

        # Portfolio Features 数量（position, unrealized_pnl, trade_duration, total_profit, cash_ratio）
        NUM_PORTFOLIO_FEATURES = 5

        # Reward constants (可通过config调整)
        # 优化版本 v2：进一步平衡，减少 Exit 主导
        INVALID_ACTION_PENALTY = -2.0       # 无效动作惩罚
        EXIT_PROFIT_BASE = 1.5              # 盈利平仓基础奖励（再降：2->1.5）
        EXIT_PROFIT_MULTIPLIER = 15         # 盈利平仓乘数（再降：20->15）
        EXIT_LOSS_BASE = -2.0               # 亏损平仓基础惩罚
        EXIT_LOSS_MULTIPLIER = 25           # 亏损平仓乘数（降低：30->25）
        ENTRY_REWARD = 0.5                  # 入场奖励
        HOLD_NEUTRAL_PENALTY = -0.05        # 空仓持有惩罚（再减：-0.1->-0.05）
        HOLD_PROFIT_BASE = 0.3              # 盈利持仓基础奖励
        HOLD_PROFIT_MULTIPLIER = 10         # 盈利持仓乘数
        HOLD_LOSS_BASE = -0.5               # 亏损持仓基础惩罚
        HOLD_LOSS_MULTIPLIER = 15           # 亏损持仓乘数

        def reset_env(self, df, prices, window_size, reward_kwargs, starting_point=True):
            """
            覆盖父类方法，更新observation_space以包含额外的Portfolio Features。
            """
            # 调用父类的reset_env
            super().reset_env(df, prices, window_size, reward_kwargs, starting_point)
            
            # 更新total_features以包含Portfolio Features
            self.total_features = self.signal_features.shape[1] + self.NUM_PORTFOLIO_FEATURES
            self.shape = (window_size, self.total_features)
            
            # 更新observation_space
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32
            )
            
            logger.info(
                f"MyRLEnv: observation_space updated to {self.shape} "
                f"(+{self.NUM_PORTFOLIO_FEATURES} portfolio features)"
            )

        def _get_observation(self):
            """
            覆盖父类方法，强制添加持仓信息到observation中。
            这样LLM就能知道当前是否有持仓，做出更合理的决策。
            
            添加的状态信息 (Portfolio Features):
            - position: 0=Short, 0.5=Neutral, 1=Long
            - unrealized_pnl: 当前未实现盈亏
            - trade_duration: 当前持仓时长（candles）
            - total_profit: 累计已实现收益
            - cash_ratio: 现金比例 (1=全现金, 0=全仓位)
            """
            # 获取原始特征窗口
            features_window = self.signal_features[
                (self._current_tick - self.window_size) : self._current_tick
            ]
            
            # 创建状态信息列
            state_info = pd.DataFrame(
                np.zeros((len(features_window), 5)),
                columns=[
                    "%-position", 
                    "%-unrealized_pnl", 
                    "%-trade_duration",
                    "%-total_profit",
                    "%-cash_ratio",
                ],
                index=features_window.index,
            )
            
            # 填充状态信息（整个窗口使用当前状态）
            state_info["%-position"] = self._position.value
            state_info["%-unrealized_pnl"] = self.get_unrealized_profit()
            state_info["%-trade_duration"] = self.get_trade_duration()
            state_info["%-total_profit"] = self._total_profit - 1.0  # 转为收益率形式
            
            # cash_ratio: Neutral时=1（全现金），Long/Short时=0（全仓位）
            # 这是简化模型，假设每次交易使用全部资金
            if self._position == Positions.Neutral:
                state_info["%-cash_ratio"] = 1.0
            else:
                state_info["%-cash_ratio"] = 0.0
            
            # 合并特征和状态信息
            features_and_state = pd.concat([features_window, state_info], axis=1)
            
            return features_and_state

        def step(self, action):
            """确保action是int类型"""
            if hasattr(action, 'item'):
                action = action.item()
            return super().step(int(action))

        def _get_exit_reward(self, current_pnl: float) -> float:
            """计算exit action的reward"""
            if current_pnl > 0:
                return self.EXIT_PROFIT_BASE + current_pnl * self.EXIT_PROFIT_MULTIPLIER
            return self.EXIT_LOSS_BASE + current_pnl * self.EXIT_LOSS_MULTIPLIER

        def _get_hold_reward(self, current_pnl: float) -> float:
            """计算hold的reward"""
            if current_pnl > 0:
                return self.HOLD_PROFIT_BASE + current_pnl * self.HOLD_PROFIT_MULTIPLIER
            return self.HOLD_LOSS_BASE + current_pnl * self.HOLD_LOSS_MULTIPLIER

        def calculate_reward(self, action: int) -> float:
            """计算reward，平衡正负信号"""
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
        """清理GPU内存"""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "llm_backbone") and self.llm_backbone is not None:
            del self.llm_backbone
            self.llm_backbone = None
        if hasattr(self, "prompt_builder"):
            self.prompt_builder = None

        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {th.cuda.memory_allocated() / 1e9:.2f} GB")

    def _check_initial_policy_distribution(
        self,
        actor: nn.Module,
        train_df: DataFrame,
        device: th.device,
        num_actions: int,
    ) -> None:
        """
        检查初始policy的action分布是否均匀。
        
        如果某个action概率过高，说明初始化有问题，可能导致policy collapse。
        """
        actor.eval()
        
        # 随机采样一些observations
        sample_size = min(50, len(train_df))
        sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
        sample_data = train_df.iloc[sample_indices].values
        
        if sample_data.ndim > 2:
            sample_data = sample_data.reshape(sample_data.shape[0], -1)
        
        with th.no_grad():
            obs_tensor = th.tensor(sample_data, dtype=th.float32).to(device)
            input_td = TensorDict({"observation": obs_tensor}, batch_size=[sample_size])
            output_td = actor(input_td)
            
            logits = output_td["logits"]
            probs = th.softmax(logits, dim=-1)
            
            # 计算平均概率分布
            avg_probs = probs.mean(dim=0).cpu().numpy()
            
            # 计算熵（衡量分布均匀程度）
            entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-8))
            max_entropy = np.log(num_actions)  # 均匀分布的熵
            
            action_names = ["Neutral", "Long_Enter", "Long_Exit", "Short_Enter", "Short_Exit"]
            logger.info("=" * 60)
            logger.info("Initial Policy Distribution Check:")
            for i, (name, p) in enumerate(zip(action_names, avg_probs, strict=False)):
                bar = "█" * int(p * 40)
                logger.info(f"  {name:12s}: {p:.4f} {bar}")
            logger.info(f"  Entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
            
            # 警告：如果某个action概率过高
            max_prob = avg_probs.max()
            if max_prob > 0.5:
                logger.warning(f"⚠️  Action {avg_probs.argmax()} has probability {max_prob:.2%} - "
                              "risk of policy collapse!")
            elif entropy < max_entropy * 0.7:
                logger.warning(f"⚠️  Low entropy ({entropy:.4f}) - distribution not uniform enough")
            else:
                logger.info("✓  Initial policy distribution looks reasonable")
            logger.info("=" * 60)
        
        actor.train()

    def _log_training_progress(
        self,
        batch_idx: int,
        reward_t: th.Tensor,
        action_t: th.Tensor,
        num_actions: int,
        avg_loss: float,
        current_lr: float | None = None,
    ) -> None:
        """Log training progress."""
        avg_reward = reward_t.mean().item()
        action_counts = [(action_t == a).sum().item() for a in range(num_actions)]
        action_names = ["N", "LE", "LX", "SE", "SX"]
        action_dist = ", ".join(
            f"{n}:{c}" for n, c in zip(action_names, action_counts, strict=True)
        )
        r_pos = (reward_t > 0).sum().item()
        r_neg = (reward_t < 0).sum().item()
        lr_str = f", LR={current_lr:.2e}" if current_lr is not None else ""
        logger.info(
            f"Batch {batch_idx}: Reward={avg_reward:.4f} (+:{r_pos}, -:{r_neg}), "
            f"Loss={avg_loss:.4f}{lr_str}, Actions=[{action_dist}]"
        )

        if th.cuda.is_available():
            mem_gb = th.cuda.memory_allocated() / 1e9
            logger.debug(f"GPU memory: {mem_gb:.2f} GB")

    def _build_networks(
        self,
        input_dim: int,
        output_dim: int,
        device: th.device,
    ) -> tuple[ProbabilisticActor, ValueOperator, LLMValueNetwork]:
        """
        构建LLM-based Actor和Critic网络。

        Args:
            input_dim: 输入观察维度
            output_dim: 动作空间大小
            device: 计算设备

        Returns:
            actor, value_module, value_net
        """
        # 从config获取超参数
        freeze_layers = self.rl_config.get("freeze_layers", 25)
        max_length = self.rl_config.get("max_prompt_length", 512)

        # 构建LLM backbone
        logger.info("Building LLM backbone...")
        self.llm_backbone = LLMBackbone(
            model_path=self.llm_model_path,
            freeze_layers=freeze_layers,
            max_length=max_length,
            dtype=th.bfloat16,  # bfloat16 数值范围更大，避免 NaN
        ).to(device)

        if th.cuda.is_available():
            logger.info(f"GPU memory after loading LLM: {th.cuda.memory_allocated() / 1e9:.2f} GB")

        # 创建共享backbone的Policy和Value网络
        policy_net = LLMPolicyNetwork(
            self.llm_backbone, output_dim, self.prompt_builder
        ).to(device)

        value_net = LLMValueNetwork(
            self.llm_backbone, self.prompt_builder
        ).to(device)

        # 构建TorchRL组件
        actor_module = TensorDictModule(
            policy_net, in_keys=["observation"], out_keys=["logits"]
        )

        action_spec = CategoricalSpec(n=output_dim, device=device)

        actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        ).to(device)

        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        ).to(device)

        return actor, value_module, value_net

    def _create_env_maker(
        self,
        train_df: DataFrame,
        prices_train: DataFrame,
        env_info: dict,
        device: th.device,
    ):
        """
        创建TorchRL collector使用的环境工厂函数。

        Args:
            train_df: 训练数据
            prices_train: 价格数据
            env_info: 环境配置信息
            device: 计算设备

        Returns:
            env_maker函数
        """
        MyRLEnv = self.MyRLEnv
        max_steps = len(train_df)

        def env_maker():
            gym_env = MyRLEnv(df=train_df, prices=prices_train, **env_info)
            gym_env = FlattenObsGymWrapper(gym_env)
            torchrl_env = GymWrapper(gym_env, device=device, categorical_action_encoding=True)
            return TransformedEnv(
                torchrl_env,
                Compose(StepCounter(max_steps=max_steps))
            )

        return env_maker

    def _compute_gae_manually(
        self,
        tensordict_data: TensorDict,
        value_net: nn.Module,
        gamma: float,
        gae_lambda: float,
        device: th.device,
    ) -> None:
        """
        手动计算 GAE (Generalized Advantage Estimation)。
        
        避免使用 TorchRL 的 GAE 模块，因为它内部使用 vmap，
        而 vmap 创建的批量张量与 LLM 的 text-based 处理不兼容。
        
        Args:
            tensordict_data: 收集的轨迹数据
            value_net: 价值网络
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            device: 计算设备
        """
        # 获取 observations 和 rewards
        obs = tensordict_data["observation"]  # [batch, obs_dim]
        next_obs = tensordict_data["next", "observation"]  # [batch, obs_dim]
        rewards = tensordict_data["next", "reward"]  # [batch, 1] or [batch]
        dones = tensordict_data["next", "done"]  # [batch, 1] or [batch]
        
        # 确保形状正确
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        
        batch_size = obs.shape[0]
        
        # 计算当前状态和下一状态的 value（逐个处理，避免 vmap）
        with th.no_grad():
            # 分批计算 value 以避免 OOM
            values = []
            next_values = []
            mini_batch = 8  # 小批量处理
            
            for i in range(0, batch_size, mini_batch):
                end_idx = min(i + mini_batch, batch_size)
                
                obs_batch = obs[i:end_idx].to(device)
                next_obs_batch = next_obs[i:end_idx].to(device)
                
                # 创建 TensorDict 并调用 value_net
                obs_td = TensorDict({"observation": obs_batch}, batch_size=[end_idx - i])
                next_obs_td = TensorDict({"observation": next_obs_batch}, batch_size=[end_idx - i])
                
                value_net(obs_td)
                value_net(next_obs_td)
                
                values.append(obs_td["state_value"].cpu())
                next_values.append(next_obs_td["state_value"].cpu())
            
            values = th.cat(values, dim=0)  # [batch, 1]
            next_values = th.cat(next_values, dim=0)  # [batch, 1]
        
        # 计算 TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        rewards = rewards.cpu()
        dones = dones.cpu().float()
        
        # 数值稳定性检查
        values = th.nan_to_num(values, nan=0.0, posinf=100.0, neginf=-100.0)
        next_values = th.nan_to_num(next_values, nan=0.0, posinf=100.0, neginf=-100.0)
        rewards = th.nan_to_num(rewards, nan=0.0, posinf=100.0, neginf=-100.0)
        
        td_errors = rewards + gamma * next_values * (1 - dones) - values
        
        # 计算 GAE: A_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}
        # 简化版本：对于单个 batch，我们假设是独立的 transitions
        # 对于更精确的 GAE，需要按轨迹处理，但这里简化处理
        advantages = td_errors.clone()
        advantages = th.clamp(advantages, min=-100.0, max=100.0)
        
        # 如果有连续轨迹信息，可以做更精确的 GAE
        # 这里使用简化版本：每个 transition 独立
        # 对于短轨迹或随机采样，这个近似是合理的
        
        # 计算 value_target = advantage + value
        value_targets = advantages + values
        
        # 标准化 advantage (保持 [batch, 1] 形状)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 写回 tensordict
        # 注意: TorchRL PPO 期望 [batch, 1] 形状，不要 squeeze
        tensordict_data["advantage"] = advantages.to(device)
        tensordict_data["value_target"] = value_targets.to(device)
        tensordict_data["state_value"] = values.to(device)

    def fit(self, data_dictionary: dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        使用LLM Backbone + PPO训练。
        实现FLAG-TRADER论文的训练流程。
        """
        # 过滤 TorchRL 内部的 NumPy deprecation warning（来自 gym.py:1171）
        warnings.filterwarnings(
            "ignore",
            message="Conversion of an array with ndim > 0 to a scalar is deprecated",
            category=DeprecationWarning,
        )
        
        self._cleanup_memory()

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        prices_train, prices_test = self.build_ohlc_price_dataframes(
            dk.data_dictionary, dk.pair, dk
        )

        self.df_raw = copy.deepcopy(train_df)
        self.set_train_and_eval_environments(data_dictionary, prices_train, prices_test, dk)

        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {device}")

        # 获取feature names用于prompt构建
        feature_names = list(train_df.columns)
        self.prompt_builder = PromptBuilder(feature_names=feature_names)

        # 创建dummy env获取维度
        env_info = self.pack_env_dict(dk.pair)
        dummy_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_info)
        dummy_env = FlattenObsGymWrapper(dummy_env)
        input_dim = dummy_env.observation_space.shape[0]
        output_dim = dummy_env.action_space.n
        dummy_env.close()

        logger.info(f"Observation dim: {input_dim}, Action dim: {output_dim}")

        # ============ 论文超参数 (Table 3) ============
        # 注意: 5e-4 对 LLM 微调太高，容易导致 NaN，降低到 2e-5
        learning_rate = self.rl_config.get("learning_rate", 2e-5)
        gamma = self.rl_config.get("gamma", 0.95)
        gae_lambda = self.rl_config.get("gae_lambda", 0.98)
        clip_coef = self.rl_config.get("clip_coef", 0.2)
        # 增大entropy系数促进探索，防止policy collapse
        # 对于5个action，0.1-0.2是更合理的范围
        ent_coef = self.rl_config.get("ent_coef", 0.15)
        vf_coef = self.rl_config.get("vf_coef", 0.5)
        ppo_epochs = self.rl_config.get("ppo_epochs", 1)
        mini_batch_size = self.rl_config.get("mini_batch_size", 32)
        frames_per_batch = self.rl_config.get("train_batch_size", 40)
        gradient_accumulation_steps = self.rl_config.get("gradient_accumulation_steps", 8)
        max_grad_norm = self.rl_config.get("max_grad_norm", 1.0)  # LLM 训练通常用 1.0

        # ============ 构建LLM-based网络 ============
        actor, value_module, value_net = self._build_networks(input_dim, output_dim, device)

        # ============ 创建环境和Collector ============
        env_maker = self._create_env_maker(train_df, prices_train, env_info, device)

        collector = SyncDataCollector(
            env_maker,
            policy=actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_timesteps,
            split_trajs=False,
            device=device,
        )

        # 注意: 不使用 TorchRL 的 GAE 模块，因为它内部使用 vmap，
        # 而 vmap 创建的批量张量与 LLM 的 text-based 处理不兼容。
        # 改用 _compute_gae_manually() 方法手动计算 advantage。

        # PPO Loss
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=value_module,
            clip_epsilon=clip_coef,
            entropy_bonus=True,
            entropy_coeff=ent_coef,
            critic_coeff=vf_coef,
            loss_critic_type="l2",
            normalize_advantage=True,
        )
        loss_module.set_keys(advantage="advantage", value_target="value_target")

        # Optimizer (增加 eps 提高数值稳定性)
        optimizer = th.optim.AdamW(
            loss_module.parameters(), 
            lr=learning_rate,
            eps=1e-5,  # 默认是 1e-8，增大可提高稳定性
            weight_decay=0.01,  # L2 正则化有助于稳定训练
        )

        # Learning rate scheduler with warmup (LLM训练需要warmup避免初期NaN)
        total_batches = total_timesteps // frames_per_batch
        warmup_batches = max(10, total_batches // 10)  # 10% warmup
        
        # Warmup scheduler: 从 0.1x 线性增加到 1.0x
        warmup_scheduler = th.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_batches,
        )
        # Decay scheduler: 从 1.0x 线性衰减到 0.1x
        decay_scheduler = th.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=total_batches - warmup_batches,
        )
        # 组合: 先warmup后decay
        scheduler = th.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_batches],
        )

        # Replay Buffer (on-policy, 无放回采样)
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            batch_size=mini_batch_size,
            sampler=SamplerWithoutReplacement(),
        )

        # ============ 训练循环 ============
        logger.info("=" * 60)
        logger.info("Starting FLAG-TRADER LLM training with PPO")
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Frames per batch: {frames_per_batch}")
        logger.info(f"PPO epochs: {ppo_epochs}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Entropy coefficient: {ent_coef}")
        logger.info("=" * 60)
        
        # ============ 初始Policy分布检查 ============
        self._check_initial_policy_distribution(actor, train_df, device, output_dim)

        start_time = time.time()

        for batch_idx, tensordict_data in enumerate(collector):
            # 手动计算 GAE (避免 TorchRL 的 vmap 问题)
            self._compute_gae_manually(
                tensordict_data=tensordict_data,
                value_net=value_module,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
            )

            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # PPO updates with gradient accumulation
            total_loss = 0
            num_updates = 0

            for epoch in range(ppo_epochs):
                optimizer.zero_grad()
                accumulated_loss = 0

                num_mini_batches = max(1, frames_per_batch // mini_batch_size)
                for step in range(num_mini_batches):
                    subdata = replay_buffer.sample()
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Gradient accumulation
                    loss_value = loss_value / gradient_accumulation_steps
                    loss_value.backward()
                    accumulated_loss += loss_value.item()

                    if (step + 1) % gradient_accumulation_steps == 0:
                        th.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()
                        total_loss += accumulated_loss
                        accumulated_loss = 0
                        num_updates += 1

                # 处理剩余的梯度
                if accumulated_loss > 0:
                    th.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += accumulated_loss
                    num_updates += 1

            scheduler.step()
            replay_buffer.empty()

            # Logging
            if batch_idx % 5 == 0:
                reward_t = tensordict_data["next", "reward"].reshape(-1)
                action_t = tensordict_data["action"].reshape(-1)
                avg_loss = total_loss / max(1, num_updates)
                current_lr = scheduler.get_last_lr()[0]

                self._log_training_progress(
                    batch_idx=batch_idx,
                    reward_t=reward_t,
                    action_t=action_t,
                    num_actions=output_dim,
                    avg_loss=avg_loss,
                    current_lr=current_lr,
                )

        collector.shutdown()
        training_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Training finished in {training_time:.2f}s ({training_time/60:.2f} min)")
        logger.info("=" * 60)

        self.model = actor

        # Cleanup training components
        del collector, loss_module, optimizer, scheduler, replay_buffer
        del value_module, value_net
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
            logger.info(f"GPU memory after cleanup: {th.cuda.memory_allocated() / 1e9:.2f} GB")

        return actor

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model) -> DataFrame:
        """LLM-based推理"""
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

        # LLM推理用较小的batch size
        batch_size = self.rl_config.get("prediction_batch_size", 32)
        raw_data = dataframe.values
        all_actions = []

        logger.info(f"Running LLM inference on {len(raw_data)} samples...")

        with th.no_grad():
            for i in range(0, len(raw_data), batch_size):
                chunk = raw_data[i : i + batch_size]
                if chunk.ndim > 2:
                    chunk = chunk.reshape(chunk.shape[0], -1)

                obs_data = th.tensor(chunk, dtype=th.float32).to(device)
                input_td = TensorDict({"observation": obs_data}, batch_size=[len(obs_data)])
                output_td = model(input_td)

                logits = output_td["logits"]
                actions = logits.argmax(dim=-1).cpu().numpy()
                all_actions.append(actions)

                if (i // batch_size) % 10 == 0:
                    logger.debug(f"Inference progress: {i}/{len(raw_data)}")

        actions = np.concatenate(all_actions)

        # Debug logging
        unique, counts = np.unique(actions, return_counts=True)
        action_dist = dict(zip(unique, counts, strict=False))
        logger.info(f"Prediction action distribution: {action_dist}")
        logger.info(f"Total predictions: {len(actions)}")

        pred_df = pd.DataFrame(
            actions,
            columns=dk.label_list,
            index=dataframe.index,
        )

        return pred_df

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs):
        """Override predict to use LLM model"""
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

        pred_df = self.rl_model_predict(
            dk.data_dictionary["prediction_features"], dk, self.model
        )
        pred_df.fillna(0, inplace=True)

        return (pred_df, dk.do_predict)

    def save(self, path: Path) -> None:
        """保存模型"""
        if self.model:
            save_path = path / "actor_state_dict.pt"
            th.save(self.model.state_dict(), save_path)
            logger.info(f"Saved FLAG-TRADER LLM model to {save_path}")

            # 也保存LLM backbone的可训练参数
            if self.llm_backbone:
                backbone_path = path / "llm_backbone_trainable.pt"
                # 注意: state_dict() 返回的是 tensor，需要对照原始参数判断 requires_grad
                trainable_names = {
                    name for name, param in self.llm_backbone.named_parameters()
                    if param.requires_grad
                }
                trainable_state = {
                    name: tensor for name, tensor in self.llm_backbone.state_dict().items()
                    if name in trainable_names
                }
                th.save(trainable_state, backbone_path)
                logger.info(f"Saved LLM backbone trainable params to {backbone_path}")
        else:
            logger.warning("No model to save.")

    def load(self, path: Path) -> None:
        """加载模型"""
        load_path = path / "actor_state_dict.pt"
        if load_path.is_file():
            logger.info(f"Loading FLAG-TRADER LLM model from {load_path}")
            if self.model is None:
                logger.warning("Model structure not initialized. Cannot load weights.")
                return

            device = th.device("cuda" if th.cuda.is_available() else "cpu")
            state_dict = th.load(load_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Could not find model file at {load_path}")

