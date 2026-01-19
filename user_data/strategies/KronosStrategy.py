"""
Kronos Strategy - 基于 Kronos 时序基础模型的交易策略

这个策略使用 Kronos 预训练模型进行价格预测，并基于预测结果生成交易信号。
策略设计极简，只需要原始 OHLCV 数据。

Action 值说明:
- 0: Hold (持有/观望)
- 1: Long_enter (做多入场)
- 3: Short_enter (做空入场)
"""

import logging
from functools import reduce

from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)


class KronosStrategy(IStrategy):
    """
    Kronos 策略 - 使用 Kronos 金融基础模型进行预测。
    
    特点:
    1. 极简特征工程 - 只需要原始 OHLCV
    2. 无需训练 - 直接使用预训练模型推理
    3. 基于规则的信号生成
    """

    # ==================== 策略参数 ====================
    
    # ROI 设置 (相对保守)
    minimal_roi = {
        "0": 0.05,      # 5% 利润即可退出
        "120": 0.025,   # 120 分钟后 2.5%
        "240": 0.01,    # 240 分钟后 1%
        "480": 0        # 480 分钟后任何利润
    }
    
    # 止损设置
    stoploss = -0.05  # 5% 止损
    
    # 时间周期 (与 Kronos 预训练数据匹配)
    timeframe = "1h"
    
    # 启动所需的最小蜡烛数量
    # 注意: Kronos lookback=400，但策略层面设置较小值让 FreqAI 有更多数据
    startup_candle_count = 450
    
    # 启用做空
    can_short = True
    
    # 使用自定义止损
    use_custom_stoploss = False
    
    # 订单类型
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    # ==================== 特征工程 ====================

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        周期扩展特征 - Kronos 不需要复杂特征，保持简单。
        """
        # 可选: 添加一些简单的技术指标作为辅助
        # 这些不是必需的，Kronos 主要依赖原始 OHLCV
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        基础特征 - 价格变化率。
        """
        # 价格变化率 (可选辅助特征)
        dataframe["%-pct_change"] = dataframe["close"].pct_change()
        
        # 成交量变化率
        dataframe["%-volume_pct"] = dataframe["volume"].pct_change()
        
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        标准特征 - 原始 OHLCV 数据 (Kronos 必需)。
        """
        # 原始价格数据 - Kronos 预测必需
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_volume"] = dataframe["volume"]
        
        # 时间特征 (可选)
        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-dayofweek"] = dataframe["date"].dt.dayofweek
        
        return dataframe

    # ==================== 目标定义 ====================

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        定义模型目标 - Kronos 返回交易信号。
        
        信号值:
        - 0: Hold (不操作)
        - 1: Long_enter (做多入场)
        - 3: Short_enter (做空入场)
        """
        # 信号占位符，Kronos 模型会在推理时填充
        dataframe["&-action"] = 0
        return dataframe

    # ==================== 指标填充 ====================

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        主指标填充函数 - 调用 FreqAI 进行预测。
        """
        pair = metadata.get("pair", "")
        logger.info(f"[KronosStrategy] populate_indicators called for {pair}")
        logger.info(f"[KronosStrategy] Before freqai.start: dataframe shape={dataframe.shape}, "
                   f"index range: {dataframe.index.min()}-{dataframe.index.max()}")
        
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        logger.info(f"[KronosStrategy] After freqai.start: dataframe shape={dataframe.shape}")
        
        # 检查 freqai.start 返回后的数据
        if "&-action" in dataframe.columns:
            action_nonzero = (dataframe["&-action"] != 0).sum()
            action_counts = dataframe["&-action"].value_counts().to_dict()
            logger.info(f"[KronosStrategy] After freqai.start: &-action non-zero count: {action_nonzero}")
            logger.info(f"[KronosStrategy] After freqai.start: &-action distribution: {action_counts}")
        if "do_predict" in dataframe.columns:
            do_predict_sum = (dataframe["do_predict"] == 1).sum()
            logger.info(f"[KronosStrategy] After freqai.start: do_predict==1 count: {do_predict_sum}")
        
        return dataframe

    # ==================== 入场逻辑 ====================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号逻辑。
        
        Kronos 返回的 action 值:
        - 1: Long_enter (做多入场)
        - 3: Short_enter (做空入场)
        """
        # Debug: 检查 DataFrame 状态
        logger.info(f"[KronosStrategy] ========== DEBUG INFO ==========")
        logger.info(f"[KronosStrategy] DataFrame shape: {dataframe.shape}")
        logger.info(f"[KronosStrategy] Columns: {list(dataframe.columns)}")
        
        # 检查 &-action 列
        if "&-action" in dataframe.columns:
            action_counts = dataframe["&-action"].value_counts().to_dict()
            action_nonzero = (dataframe["&-action"] != 0).sum()
            logger.info(f"[KronosStrategy] &-action distribution: {action_counts}")
            logger.info(f"[KronosStrategy] &-action non-zero count: {action_nonzero}")
            logger.info(f"[KronosStrategy] &-action dtype: {dataframe['&-action'].dtype}")
            # 打印前几个非零值
            nonzero_mask = dataframe["&-action"] != 0
            if nonzero_mask.any():
                logger.info(f"[KronosStrategy] &-action non-zero samples: {dataframe.loc[nonzero_mask, '&-action'].head(10).tolist()}")
        else:
            logger.warning("[KronosStrategy] &-action column NOT FOUND!")
            
        # 检查 do_predict 列
        if "do_predict" in dataframe.columns:
            do_predict_sum = (dataframe["do_predict"] == 1).sum()
            logger.info(f"[KronosStrategy] do_predict == 1 count: {do_predict_sum}/{len(dataframe)}")
            logger.info(f"[KronosStrategy] do_predict dtype: {dataframe['do_predict'].dtype}")
            logger.info(f"[KronosStrategy] do_predict unique: {dataframe['do_predict'].unique()}")
        else:
            logger.warning("[KronosStrategy] do_predict column NOT FOUND!")
            
        logger.info(f"[KronosStrategy] ================================")

        # 做多入场: action == 1
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 1,
        ]
        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"]
            ] = (1, "kronos_long")

        # 做空入场: action == 3
        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 3,
        ]
        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"]
            ] = (1, "kronos_short")

        return dataframe

    # ==================== 出场逻辑 ====================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号逻辑。
        
        Kronos 策略使用 ROI 和止损进行出场，不使用信号出场。
        如果需要信号出场，可以扩展 action 值 (2=Long_exit, 4=Short_exit)。
        """
        # 当前版本使用 ROI/止损出场，不设置信号出场
        # 如果 Kronos 预测反转，可以在这里添加出场逻辑
        
        # 可选: 当信号变为 Hold (0) 时出场
        # exit_long_conditions = [
        #     dataframe["do_predict"] == 1,
        #     dataframe["&-action"] == 0,
        # ]
        # if exit_long_conditions:
        #     dataframe.loc[
        #         reduce(lambda x, y: x & y, exit_long_conditions),
        #         "exit_long"
        #     ] = 1
        
        return dataframe

    # ==================== 可选: 自定义止损 ====================

    def custom_stoploss(
        self, 
        pair: str, 
        trade, 
        current_time, 
        current_rate: float,
        current_profit: float, 
        **kwargs
    ) -> float:
        """
        自定义止损逻辑 (可选)。
        
        可以基于 Kronos 预测动态调整止损。
        """
        # 默认使用固定止损
        return self.stoplsh
