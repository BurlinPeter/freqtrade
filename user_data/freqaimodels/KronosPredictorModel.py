"""
Kronos FreqAI Model - Zero-Shot Time Series Prediction

基于 Kronos 金融 K 线基础模型的 FreqAI 集成。
不进行强化学习训练，直接使用预训练模型进行推理。

参考: https://github.com/shiyu-coder/Kronos
"""

import gc
import logging
import sys
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel


logger = logging.getLogger(__name__)


class KronosPredictorModel(IFreqaiModel):
    """
    Kronos 预测模型 - 使用预训练的金融时序基础模型进行推理。
    
    核心逻辑:
    1. fit() 方法仅加载模型，不进行训练
    2. predict() 方法使用 Kronos 进行未来价格预测
    3. 基于预测结果生成交易信号 (0=Hold, 1=Long, 3=Short)
    
    配置参数 (在 config.json 的 freqai.model_training_parameters 中设置):
    - kronos_model_path: Kronos 模型路径
    - tokenizer_path: Tokenizer 路径
    - lookback: 回看 K 线数量 (默认 400)
    - pred_len: 预测未来 K 线数量 (默认 24)
    - long_threshold: 做多阈值 (默认 0.015, 即 1.5%)
    - short_threshold: 做空阈值 (默认 -0.015, 即 -1.5%)
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # 从配置中读取 Kronos 参数
        self.kronos_params = self.model_training_parameters
        
        # 模型路径
        self.kronos_model_path = self.kronos_params.get(
            "kronos_model_path", 
            "user_data/models/kronos/Kronos-small"
        )
        self.tokenizer_path = self.kronos_params.get(
            "tokenizer_path",
            "user_data/models/kronos/Kronos-Tokenizer-base"
        )
        
        # 预测参数
        self.lookback = self.kronos_params.get("lookback", 400)
        self.pred_len = self.kronos_params.get("pred_len", 24)
        self.max_context = self.kronos_params.get("max_context", 512)
        
        # 信号生成阈值
        self.long_threshold = self.kronos_params.get("long_threshold", 0.015)
        self.short_threshold = self.kronos_params.get("short_threshold", -0.015)
        self.bullish_ratio_threshold = self.kronos_params.get("bullish_ratio_threshold", 0.5)
        
        # 采样参数
        self.temperature = self.kronos_params.get("temperature", 1.0)
        self.top_p = self.kronos_params.get("top_p", 0.9)
        self.sample_count = self.kronos_params.get("sample_count", 1)
        
        # 回测加速参数
        self.predict_interval = self.kronos_params.get("predict_interval", 1)  # 每 N 个样本预测一次
        self.batch_size = self.kronos_params.get("batch_size", 32)  # 批量预测大小
        
        # GPU 显存限制 (0.0-1.0，表示最多使用多少比例的显存)
        self.gpu_memory_fraction = self.kronos_params.get("gpu_memory_fraction", 0.3)  # 默认最多用 30%
        
        # Kronos 组件 (延迟加载)
        self.kronos_model = None
        self.tokenizer = None
        self.predictor = None
        
        # 历史数据缓存 - 用于在 predict() 中补充 lookback 数据
        # key: pair, value: {"ohlcv": DataFrame, "timestamps": Series}
        self.history_cache: dict[str, dict] = {}
        
        # 预测结果缓存 - 用于绕过 FreqAI 回测框架的问题
        # key: (pair, index), value: (action, do_predict)
        self.prediction_cache: dict[tuple, tuple] = {}
        
        logger.info(f"[Kronos] Initialized with lookback={self.lookback}, pred_len={self.pred_len}")
        logger.info(f"[Kronos] Thresholds: long={self.long_threshold}, short={self.short_threshold}")
        logger.info(f"[Kronos] Batch size: {self.batch_size}, predict interval: {self.predict_interval}")

    def _load_kronos(self) -> None:
        """延迟加载 Kronos 模型和分词器"""
        if self.predictor is not None:
            logger.debug("[Kronos] Model already loaded, skipping...")
            return
            
        logger.info("[Kronos] Loading model and tokenizer...")
        logger.info(f"[Kronos] Model path: {self.kronos_model_path}")
        logger.info(f"[Kronos] Tokenizer path: {self.tokenizer_path}")
        
        # 添加 kronos_lib 到 Python 路径
        kronos_lib_path = Path(__file__).parent / "kronos_lib"
        if str(kronos_lib_path) not in sys.path:
            sys.path.insert(0, str(kronos_lib_path))
        
        try:
            # 尝试从本地 kronos_lib 导入
            from model import Kronos, KronosPredictor, KronosTokenizer
            
            # 加载分词器
            logger.info(f"[Kronos] Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_path)
            
            # 加载模型
            logger.info(f"[Kronos] Loading model from {self.kronos_model_path}")
            self.kronos_model = Kronos.from_pretrained(self.kronos_model_path)
            
            # 移动到 GPU (如果可用)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 限制 GPU 显存使用
            if torch.cuda.is_available():
                # 设置显存分配上限
                torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction, device=0)
                # 清理 GPU 缓存
                torch.cuda.empty_cache()
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                max_mem = total_mem * self.gpu_memory_fraction
                logger.info(f"[Kronos] GPU memory limited to {self.gpu_memory_fraction*100:.0f}% "
                           f"({max_mem:.1f}GB / {total_mem:.1f}GB)")
            
            self.kronos_model.to(device)
            self.kronos_model.eval()
            
            # 创建预测器
            self.predictor = KronosPredictor(
                self.kronos_model, 
                self.tokenizer, 
                max_context=self.max_context
            )
            
            logger.info(f"[Kronos] Model loaded successfully on {device}")
            
        except ImportError as e:
            logger.error(f"[Kronos] Failed to import Kronos library: {e}")
            logger.error("[Kronos] Please follow the setup instructions in kronos_lib/README.md")
            raise
        except Exception as e:
            logger.error(f"[Kronos] Failed to load model: {e}")
            raise

    def train(
        self, unfiltered_df: DataFrame, pair: str, dk: FreqaiDataKitchen, **kwargs
    ) -> Any:
        """
        训练方法 - 对于 Kronos 只需要加载模型，不进行实际训练。
        
        :param unfiltered_df: 完整的训练数据
        :param pair: 交易对
        :param dk: FreqAI 数据厨房
        :return: 加载的模型实例
        """
        logger.info(f"-------------------- Loading Kronos for {pair} --------------------")
        start_time = time()
        
        # 加载 Kronos 模型
        self._load_kronos()
        
        # 保存历史 OHLCV 数据用于后续预测
        # FreqAI 在 predict() 时可能不会传递足够的历史数据
        self._cache_history(unfiltered_df, pair)
        
        # Kronos 是预训练模型，不需要复杂的特征管道
        # 设置简单的空管道以满足 FreqAI 接口要求
        dk.feature_pipeline = self.define_data_pipeline(threads=dk.thread_count)
        dk.label_pipeline = self.define_label_pipeline(threads=dk.thread_count)
        
        # 过滤特征 (仅用于获取特征列表)
        features_filtered, labels_filtered = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            dk.label_list,
            training_filter=True,
        )
        
        # 简单拟合管道以初始化 (Kronos 实际不使用这些)
        dd = dk.make_train_test_datasets(features_filtered, labels_filtered)
        
        end_time = time()
        logger.info(
            f"-------------------- Kronos loaded for {pair} "
            f"({end_time - start_time:.2f} secs) --------------------"
        )
        
        return self.predictor
    
    def _cache_history(self, df: DataFrame, pair: str) -> None:
        """
        缓存历史 OHLCV 数据。
        
        :param df: 包含 OHLCV 数据的 DataFrame
        :param pair: 交易对
        """
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"[Kronos] Cannot cache history for {pair}: missing OHLCV columns")
            return
        
        # 提取 OHLCV 数据
        ohlcv_df = df[required_cols].copy()
        if "volume" in df.columns:
            ohlcv_df["volume"] = df["volume"]
        else:
            ohlcv_df["volume"] = 0
        
        # 提取时间戳
        if "date" in df.columns:
            timestamps = pd.to_datetime(df["date"]).reset_index(drop=True)
        else:
            timestamps = pd.Series(df.index)
        
        # 保存到缓存
        self.history_cache[pair] = {
            "ohlcv": ohlcv_df.reset_index(drop=True),
            "timestamps": timestamps,
            "last_index": df.index[-1] if hasattr(df.index, '__len__') else len(df) - 1
        }
        
        logger.info(f"[Kronos] Cached {len(ohlcv_df)} candles of history for {pair}")
    
    def _get_ohlcv_with_history(self, df: DataFrame, pair: str | None) -> tuple[DataFrame, pd.Series]:
        """
        获取带有足够历史数据的 OHLCV DataFrame。
        
        如果 df 的长度不足 lookback，尝试从缓存中补充历史数据。
        
        :param df: 当前预测数据
        :param pair: 交易对
        :return: (ohlcv_df, timestamps)
        """
        required_cols = ["open", "high", "low", "close"]
        
        # 提取当前数据的 OHLCV
        current_ohlcv = df[required_cols].copy()
        if "volume" in df.columns:
            current_ohlcv["volume"] = df["volume"]
        else:
            current_ohlcv["volume"] = 0
        
        # 提取时间戳
        if "date" in df.columns:
            current_timestamps = pd.to_datetime(df["date"]).reset_index(drop=True)
        else:
            current_timestamps = pd.Series(pd.date_range(start="2020-01-01", periods=len(df), freq="1h"))
        
        # 检查是否需要补充历史数据
        if len(current_ohlcv) >= self.lookback + 100:  # 有足够的数据
            return current_ohlcv.reset_index(drop=True), current_timestamps
        
        # 尝试从缓存中获取历史数据
        if pair and pair in self.history_cache:
            cached = self.history_cache[pair]
            cached_ohlcv = cached["ohlcv"]
            cached_timestamps = cached["timestamps"]
            
            # 找到当前数据在缓存中的起始位置（基于时间戳匹配）
            try:
                first_current_time = current_timestamps.iloc[0]
                last_current_time = current_timestamps.iloc[-1]
                
                # 确保时间戳格式一致 (都转换为 datetime64[ns])
                first_current_time = pd.Timestamp(first_current_time)
                last_current_time = pd.Timestamp(last_current_time)
                cached_ts_normalized = pd.to_datetime(cached_timestamps)
                
                logger.debug(f"[Kronos] Trying to match: first={first_current_time}, last={last_current_time}")
                logger.debug(f"[Kronos] Cached range: {cached_ts_normalized.iloc[0]} to {cached_ts_normalized.iloc[-1]}")
                
                # 方法1: 直接时间戳匹配
                match_mask = cached_ts_normalized == first_current_time
                match_idx = None
                
                if match_mask.any():
                    match_idx = match_mask.idxmax()
                    logger.info(f"[Kronos] Found exact timestamp match at index {match_idx}")
                else:
                    # 方法2: 如果 df 的数据时间范围在缓存范围内，使用最近时间匹配
                    if cached_ts_normalized.iloc[0] <= first_current_time <= cached_ts_normalized.iloc[-1]:
                        # 找最接近的时间戳
                        time_diffs = abs(cached_ts_normalized - first_current_time)
                        match_idx = time_diffs.idxmin()
                        logger.info(f"[Kronos] Found nearest timestamp match at index {match_idx}, "
                                   f"diff={(cached_ts_normalized.iloc[match_idx] - first_current_time)}")
                    else:
                        # 方法3: 如果当前数据在缓存之后，直接追加缓存中的所有历史数据
                        if first_current_time > cached_ts_normalized.iloc[-1]:
                            logger.info(f"[Kronos] Current data is after cache, prepending all cached history")
                            # 直接使用缓存的最后 lookback+50 条数据作为历史
                            history_start = max(0, len(cached_ohlcv) - self.lookback - 50)
                            history_ohlcv = cached_ohlcv.iloc[history_start:].reset_index(drop=True)
                            history_timestamps = cached_ts_normalized.iloc[history_start:].reset_index(drop=True)
                            
                            # 合并历史数据和当前数据
                            combined_ohlcv = pd.concat([history_ohlcv, current_ohlcv.reset_index(drop=True)], ignore_index=True)
                            combined_timestamps = pd.concat([history_timestamps, current_timestamps], ignore_index=True)
                            
                            logger.info(f"[Kronos] Prepended {len(history_ohlcv)} candles from cache, total: {len(combined_ohlcv)}")
                            return combined_ohlcv, combined_timestamps
                
                if match_idx is not None:
                    # 从缓存中提取需要的历史数据
                    history_start = max(0, match_idx - self.lookback - 50)
                    history_ohlcv = cached_ohlcv.iloc[history_start:match_idx].reset_index(drop=True)
                    history_timestamps = cached_ts_normalized.iloc[history_start:match_idx].reset_index(drop=True)
                    
                    if len(history_ohlcv) > 0:
                        # 合并历史数据和当前数据
                        combined_ohlcv = pd.concat([history_ohlcv, current_ohlcv.reset_index(drop=True)], ignore_index=True)
                        combined_timestamps = pd.concat([history_timestamps, current_timestamps], ignore_index=True)
                        
                        logger.info(f"[Kronos] Added {len(history_ohlcv)} candles from cache, total: {len(combined_ohlcv)}")
                        return combined_ohlcv, combined_timestamps
                    else:
                        logger.warning(f"[Kronos] match_idx={match_idx} but no history before it")
                        
            except Exception as e:
                logger.warning(f"[Kronos] Failed to merge cached history: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        # 无法补充历史数据，返回原始数据
        logger.warning(f"[Kronos] No cached history available for {pair}, using original data ({len(current_ohlcv)} candles)")
        return current_ohlcv.reset_index(drop=True), current_timestamps

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        fit 方法 - 被 train() 调用，这里只返回已加载的预测器。
        """
        return self.predictor

    def predict(
        self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[DataFrame, npt.NDArray[np.int_]]:
        """
        预测方法 - 使用 Kronos 进行价格预测并生成交易信号。
        
        :param unfiltered_df: 完整的预测数据
        :param dk: FreqAI 数据厨房
        :return: (预测结果 DataFrame, do_predict 数组)
        """
        # 确保模型已加载
        self._load_kronos()
        
        # 获取特征列表
        dk.find_features(unfiltered_df)
        
        # 过滤特征 (不应用 feature_pipeline，Kronos 直接使用原始 OHLCV)
        dk.data_dictionary["prediction_features"], _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        
        n_samples = len(dk.data_dictionary["prediction_features"])
        prediction_index = dk.data_dictionary["prediction_features"].index
        
        logger.info(f"[Kronos] predict() called: n_samples={n_samples}, "
                   f"unfiltered_df len={len(unfiltered_df)}, label_list={dk.label_list}")
        
        # 更多调试信息
        logger.info(f"[Kronos] unfiltered_df index: first={unfiltered_df.index[0]}, last={unfiltered_df.index[-1]}")
        logger.info(f"[Kronos] unfiltered_df columns: {list(unfiltered_df.columns)[:10]}...")
        if "date" in unfiltered_df.columns:
            logger.info(f"[Kronos] unfiltered_df date range: {unfiltered_df['date'].iloc[0]} to {unfiltered_df['date'].iloc[-1]}")
        
        # Kronos 不需要 FreqAI 的特征管道，直接使用原始数据
        # 设置所有样本为有效预测
        dk.do_predict = np.ones(n_samples, dtype=np.int_)
        
        # 提取 OHLCV 数据进行 Kronos 预测
        predictions = self._kronos_predict(unfiltered_df, dk)
        
        # 创建预测 DataFrame
        # 重要: 必须使用 reset_index 后的索引 (0, 1, 2, ...)
        # 因为 FreqAI 的 get_predictions_to_append() 会对 dataframe_backtest 做 reset_index
        # 然后用 pd.concat(axis=1) 合并，此时索引必须对齐
        pred_df = DataFrame(
            predictions, 
            columns=dk.label_list
            # 不指定 index，使用默认的 0, 1, 2, ... 索引
        )
        
        logger.info(f"[Kronos] pred_df shape={pred_df.shape}, do_predict sum={dk.do_predict.sum()}")
        logger.info(f"[Kronos] pred_df value_counts: {pred_df[dk.label_list[0]].value_counts().to_dict()}")
        logger.info(f"[Kronos] pred_df index: first={pred_df.index[0]}, last={pred_df.index[-1]} (should be 0-based)")
        logger.info(f"[Kronos] prediction_features index: first={prediction_index[0]}, last={prediction_index[-1]}")
        
        # 打印更多调试信息
        logger.info(f"[Kronos] pred_df non-zero action count: {(pred_df[dk.label_list[0]] != 0).sum()}")
        logger.info(f"[Kronos] pred_df action==1 count: {(pred_df[dk.label_list[0]] == 1).sum()}")
        logger.info(f"[Kronos] pred_df action==3 count: {(pred_df[dk.label_list[0]] == 3).sum()}")
        
        # 打印 dk 相关信息
        logger.info(f"[Kronos] dk.pair: {dk.pair if hasattr(dk, 'pair') else 'NOT SET'}")
        logger.info(f"[Kronos] dk.label_list: {dk.label_list}")
        
        # 验证索引对齐
        logger.info(f"[Kronos] ✓ pred_df uses 0-based index for proper alignment with FreqAI")
        
        return (pred_df, dk.do_predict)

    def _kronos_predict(self, df: DataFrame, dk: FreqaiDataKitchen) -> np.ndarray:
        """
        使用 Kronos 批量预测并生成交易信号。
        
        :param df: 包含 OHLCV 数据的 DataFrame
        :param dk: FreqAI 数据厨房
        :return: 交易信号数组 (0=Hold, 1=Long, 3=Short)
        """
        n_samples = len(dk.data_dictionary["prediction_features"])
        signals = np.zeros(n_samples, dtype=np.int32)
        
        # 检查是否有必要的 OHLCV 列
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            logger.warning("[Kronos] Missing OHLCV columns, returning neutral signals")
            return signals
        
        # 获取当前交易对
        pair = dk.pair if hasattr(dk, 'pair') else (next(iter(self.history_cache.keys())) if self.history_cache else None)
        
        # 获取 OHLCV 数据 - 尝试使用缓存的历史数据补充
        ohlcv_df, timestamps = self._get_ohlcv_with_history(df, pair)
        
        if self.predictor is None:
            logger.error("[Kronos] Predictor not loaded! Returning all zeros.")
            return signals
        
        # 计算预测索引时需要考虑历史数据的偏移
        # history_offset 表示在 ohlcv_df 中，df 数据开始的位置
        history_offset = len(ohlcv_df) - len(df)
        
        # 收集需要预测的样本索引
        predict_indices = []
        for i in range(n_samples):
            if self.predict_interval > 1 and i % self.predict_interval != 0:
                continue
            # pred_idx 是在 ohlcv_df 中的绝对位置
            pred_idx = history_offset + len(df) - n_samples + i
            # 只处理有足够历史数据的样本 (使用固定的 lookback 长度)
            if pred_idx >= self.lookback:
                predict_indices.append(i)
        
        logger.info(f"[Kronos] Total samples: {n_samples}, valid for batch prediction: {len(predict_indices)}")
        logger.info(f"[Kronos] DataFrame length: {len(df)}, with history: {len(ohlcv_df)}, offset: {history_offset}")
        logger.info(f"[Kronos] Batch size: {self.batch_size}, predict interval: {self.predict_interval}")
        
        if len(predict_indices) == 0:
            logger.warning("[Kronos] No samples with enough history, returning all zeros")
            return signals
        
        # 计算时间间隔
        if len(timestamps) >= 2:
            time_delta = timestamps.iloc[1] - timestamps.iloc[0]
        else:
            time_delta = pd.Timedelta(hours=1)
        
        # 批量预测
        total_batches = (len(predict_indices) + self.batch_size - 1) // self.batch_size
        batch_results = {}  # {sample_idx: signal}
        
        start_time = time()
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(predict_indices))
            batch_sample_indices = predict_indices[batch_start:batch_end]
            
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                elapsed = time() - start_time
                logger.info(f"[Kronos] Batch {batch_idx + 1}/{total_batches} "
                           f"({100 * (batch_idx + 1) / total_batches:.1f}%, elapsed: {elapsed:.1f}s)")
                # 定期清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            try:
                # 准备批量数据 - 所有样本使用相同的 lookback 长度
                df_list = []
                x_timestamp_list = []
                y_timestamp_list = []
                current_prices = []
                
                for sample_i in batch_sample_indices:
                    # pred_idx 是在 ohlcv_df 中的绝对位置
                    pred_idx = history_offset + len(df) - n_samples + sample_i
                    start_idx = pred_idx - self.lookback
                    
                    # 历史数据 (固定 lookback+1 长度)
                    history_df = ohlcv_df.iloc[start_idx:pred_idx + 1].copy()
                    df_list.append(history_df)
                    current_prices.append(history_df["close"].iloc[-1])
                    
                    # 历史时间戳
                    x_ts = pd.Series(timestamps.iloc[start_idx:pred_idx + 1].values).reset_index(drop=True)
                    x_timestamp_list.append(x_ts)
                    
                    # 未来时间戳
                    y_ts = pd.Series(pd.date_range(
                        start=x_ts.iloc[-1] + time_delta,
                        periods=self.pred_len,
                        freq=time_delta
                    )).reset_index(drop=True)
                    y_timestamp_list.append(y_ts)
                
                # 调用批量预测 (禁用梯度计算以节省内存)
                with torch.no_grad():
                    pred_dfs = self.predictor.predict_batch(
                        df_list=df_list,
                        x_timestamp_list=x_timestamp_list,
                        y_timestamp_list=y_timestamp_list,
                        pred_len=self.pred_len,
                        T=self.temperature,
                        top_p=self.top_p,
                        sample_count=self.sample_count,
                        verbose=False
                    )
                
                # 生成信号
                for j, sample_i in enumerate(batch_sample_indices):
                    signal = self._generate_signal(
                        current_price=current_prices[j],
                        pred_df=pred_dfs[j],
                        log_sample=(batch_idx == 0 and j < 2)
                    )
                    batch_results[sample_i] = signal
                
                # 清理批次数据，释放内存
                del df_list, x_timestamp_list, y_timestamp_list, current_prices, pred_dfs
                    
            except Exception as e:
                if batch_idx < 3:
                    logger.error(f"[Kronos] Batch {batch_idx} failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                # 批量失败时，将这批样本标记为 Hold
                for sample_i in batch_sample_indices:
                    batch_results[sample_i] = 0
        
        # 填充结果到 signals 数组
        last_signal = 0
        for i in range(n_samples):
            if i in batch_results:
                signals[i] = batch_results[i]
                last_signal = batch_results[i]
            else:
                # 跳过的样本或历史不足的样本，使用上一个信号
                signals[i] = last_signal
        
        # 最终清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 统计信号分布
        unique, counts = np.unique(signals, return_counts=True)
        signal_dist = dict(zip(unique, counts, strict=False))
        elapsed_total = time() - start_time
        logger.info(f"[Kronos] Completed in {elapsed_total:.1f}s, signal distribution: {signal_dist}")
        
        return signals

    def _generate_signal(self, current_price: float, pred_df: DataFrame, log_sample: bool = False) -> int:
        """
        根据 Kronos 预测结果生成交易信号。
        
        信号逻辑:
        1. 看涨 (Long=1): 预测平均价涨幅 > long_threshold 且阳线比例 > bullish_ratio_threshold
        2. 看跌 (Short=3): 预测平均价跌幅 < short_threshold 且阴线比例 > bullish_ratio_threshold
        3. 持平 (Hold=0): 其他情况
        
        :param current_price: 当前价格
        :param pred_df: Kronos 预测的未来 OHLCV DataFrame
        :param log_sample: 是否打印调试日志
        :return: 交易信号 (0, 1, 或 3)
        """
        if pred_df is None or len(pred_df) == 0:
            if log_sample:
                logger.warning(f"[Kronos DEBUG] _generate_signal: pred_df is None or empty!")
            return 0
        
        # 计算预测平均价
        pred_avg_price = pred_df["close"].mean()
        pred_final_price = pred_df["close"].iloc[-1]
        
        # 计算价格变化率
        price_change_pct = (pred_avg_price - current_price) / current_price
        final_change_pct = (pred_final_price - current_price) / current_price
        
        # 计算阳线/阴线比例
        bullish_candles = (pred_df["close"] > pred_df["open"]).sum()
        bearish_candles = (pred_df["close"] < pred_df["open"]).sum()
        total_candles = len(pred_df)
        
        bullish_ratio = bullish_candles / total_candles if total_candles > 0 else 0.5
        bearish_ratio = bearish_candles / total_candles if total_candles > 0 else 0.5
        
        # Debug logging for first few samples
        if log_sample:
            logger.info(f"[Kronos DEBUG] Signal calc: current={current_price:.2f}, pred_avg={pred_avg_price:.2f}, "
                       f"pred_final={pred_final_price:.2f}")
            logger.info(f"[Kronos DEBUG] price_change%={price_change_pct*100:.4f}% (threshold: >{self.long_threshold*100:.2f}% for long, <{self.short_threshold*100:.2f}% for short)")
            logger.info(f"[Kronos DEBUG] bullish_ratio={bullish_ratio:.2f}, bearish_ratio={bearish_ratio:.2f} (threshold: >{self.bullish_ratio_threshold:.2f})")
            
            # 解释为什么生成特定信号
            if price_change_pct > self.long_threshold:
                if bullish_ratio > self.bullish_ratio_threshold:
                    logger.info(f"[Kronos DEBUG] -> LONG signal: change% > threshold AND bullish > threshold")
                else:
                    logger.info(f"[Kronos DEBUG] -> HOLD: change% > threshold BUT bullish_ratio {bullish_ratio:.2f} <= {self.bullish_ratio_threshold}")
            elif price_change_pct < self.short_threshold:
                if bearish_ratio > self.bullish_ratio_threshold:
                    logger.info(f"[Kronos DEBUG] -> SHORT signal: change% < threshold AND bearish > threshold")
                else:
                    logger.info(f"[Kronos DEBUG] -> HOLD: change% < threshold BUT bearish_ratio {bearish_ratio:.2f} <= {self.bullish_ratio_threshold}")
            else:
                logger.info(f"[Kronos DEBUG] -> HOLD: change% {price_change_pct*100:.4f}% is within [{self.short_threshold*100:.2f}%, {self.long_threshold*100:.2f}%]")
        
        # 生成信号
        if price_change_pct > self.long_threshold and bullish_ratio > self.bullish_ratio_threshold:
            return 1  # Long
        elif price_change_pct < self.short_threshold and bearish_ratio > self.bullish_ratio_threshold:
            return 3  # Short
        else:
            return 0  # Hold

    def get_cached_prediction(self, pair: str, index: int) -> tuple[int, int]:
        """
        获取缓存的预测结果。
        
        :param pair: 交易对
        :param index: DataFrame 索引
        :return: (action, do_predict) 或 (0, 0) 如果没有缓存
        """
        return self.prediction_cache.get((pair, index), (0, 0))
    
    def apply_cached_predictions(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        将缓存的预测结果应用到 DataFrame。
        
        :param dataframe: 策略 DataFrame
        :param pair: 交易对
        :return: 更新后的 DataFrame
        """
        logger.info(f"[Kronos] apply_cached_predictions called for {pair}")
        logger.info(f"[Kronos] DataFrame shape: {dataframe.shape}, index range: {dataframe.index.min()} - {dataframe.index.max()}")
        logger.info(f"[Kronos] Cached predictions count: {len(self.prediction_cache)}")
        
        # 获取该交易对的所有缓存键
        pair_keys = [(p, idx) for (p, idx) in self.prediction_cache.keys() if p == pair]
        if pair_keys:
            cached_indices = [idx for (_, idx) in pair_keys]
            logger.info(f"[Kronos] Cached indices for {pair}: min={min(cached_indices)}, max={max(cached_indices)}, count={len(cached_indices)}")
        else:
            logger.warning(f"[Kronos] No cached predictions found for pair: {pair}")
            # 打印所有缓存的 pair
            cached_pairs = set(p for (p, _) in self.prediction_cache.keys())
            logger.warning(f"[Kronos] Cached pairs: {cached_pairs}")
        
        applied = 0
        for idx in dataframe.index:
            if (pair, idx) in self.prediction_cache:
                action, do_pred = self.prediction_cache[(pair, idx)]
                dataframe.loc[idx, "&-action"] = action
                dataframe.loc[idx, "do_predict"] = do_pred
                applied += 1
        
        logger.info(f"[Kronos] Applied {applied} cached predictions to DataFrame for {pair}")
        
        # 如果没有应用任何缓存，打印更多调试信息
        if applied == 0 and pair_keys:
            df_indices = set(dataframe.index.tolist())
            cached_idx_set = set(cached_indices)
            overlap = df_indices & cached_idx_set
            logger.warning(f"[Kronos] Index overlap between DataFrame and cache: {len(overlap)}")
            logger.warning(f"[Kronos] DataFrame index sample: {list(dataframe.index[:10])}")
            logger.warning(f"[Kronos] Cached index sample: {cached_indices[:10] if cached_indices else []}")
            
        return dataframe

    def define_data_pipeline(self, threads: int = 1):
        """
        定义数据预处理管道。
        
        注意: Kronos 是零样本预训练模型，直接使用原始 OHLCV 数据进行推理。
        不需要 DissimilarityIndex 等异常值过滤器，否则会将所有预测样本标记为无效。
        """
        from datasieve.pipeline import Pipeline
        from datasieve.transforms import SKLearnWrapper
        from sklearn.preprocessing import MinMaxScaler
        
        # 简化管道：仅做基本的缩放，不进行异常值过滤
        # DissimilarityIndex 会错误地将所有样本标记为 do_predict=0
        feature_pipeline = Pipeline([
            ("scaler", SKLearnWrapper(MinMaxScaler(feature_range=(-1, 1)))),
        ])
        
        return feature_pipeline

    def define_label_pipeline(self, threads: int = 1):
        """定义标签预处理管道"""
        from datasieve.pipeline import Pipeline
        
        label_pipeline = Pipeline([])
        return label_pipeline
