import logging
from functools import reduce

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import IStrategy


logger = logging.getLogger(__name__)

class FlagTraderStrategy(IStrategy):
    """
    FlagTrader 策略模板。
    这个策略文件主要用于定义特征(Features), 这些特征将被 FlagTraderModel(LLM)用于决策。
    """

    # 策略参数
    minimal_roi = {"0": 0.1, "240": -1}
    stoploss = -0.10
    timeframe = "1d"

    # 启用 FreqAI
    can_short = True

    # --------------------------------------
    # 1. 特征工程 (Feature Engineering)
    # --------------------------------------

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        在此处定义随周期 (period) 变化的特征。
        FreqAI 会根据 config.json 中的 indicator_periods_candles 自动扩展这些特征。
        例如: 如果 indicator_periods_candles = [10, 20],
        这里定义的 %-rsi-period 会自动生成 %-rsi-10 和 %-rsi-20 两列。

        注意: 所有特征列名必须以 '%' 开头。
        """
        # RSI 指标
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

        # EMA 均线
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        # 布林带宽度
        bollinger = ta.BBANDS(dataframe, timeperiod=period)
        dataframe["%-bb_width-period"] = (
            (bollinger["upperband"] - bollinger["lowerband"]) / bollinger["middleband"]
        )

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        在此处定义不随周期变化的基础特征。
        """
        # 价格变化率 (Returns)
        dataframe["%-pct-change"] = dataframe["close"].pct_change()

        # 成交量
        dataframe["%-volume"] = dataframe["volume"]

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        在此处定义其他标准特征, 例如时间特征。
        """
        # 小时和星期几(有助于模型学习市场的时间规律)
        dataframe["%-hour"] = dataframe["date"].dt.hour
        dataframe["%-dayofweek"] = dataframe["date"].dt.dayofweek

        # 必须添加以下价格列供 RL 环境使用
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        return dataframe

    # --------------------------------------
    # 2. 目标定义 (Target Definition)
    # --------------------------------------

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        定义模型训练的目标(Labels)。
        对于 RL(强化学习)模型, 这个目标列是占位符, agent 会在推理时填充 action。
        注意: 所有目标列名必须以 '&' 开头。
        """
        # 对于 RL 模型，&-action 是占位符，agent 会在推理时返回 action 值
        # Action 值: 0=Neutral, 1=Long_enter, 2=Long_exit, 3=Short_enter, 4=Short_exit
        dataframe["&-action"] = 0
        return dataframe

    # --------------------------------------
    # 3. 策略逻辑 (Strategy Logic)
    # --------------------------------------

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        主指标填充函数。
        对于 FreqAI 策略, 必须调用 self.freqai.start() 来启动特征工程和模型推理。
        """
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        进场逻辑。
        RL 模型返回的 action 值:
        - 0: Neutral (不操作)
        - 1: Long_enter (做多入场)
        - 2: Long_exit (做多出场)
        - 3: Short_enter (做空入场)
        - 4: Short_exit (做空出场)
        """
        # Debug: 检查 &-action 列的分布
        if "&-action" in dataframe.columns:
            action_counts = dataframe["&-action"].value_counts().to_dict()
            do_predict_sum = (dataframe["do_predict"] == 1).sum()
            logger.info(f"[Strategy] &-action distribution: {action_counts}")
            logger.info(f"[Strategy] do_predict==1 count: {do_predict_sum}")
            logger.info(f"[Strategy] Total rows: {len(dataframe)}")

        # 做多入场: action == 1 (Long_enter)
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 1,
        ]
        if enter_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_long_conditions),
                ["enter_long", "enter_tag"]
            ] = (1, "long")

        # 做空入场: action == 3 (Short_enter)
        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 3,
        ]
        if enter_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, enter_short_conditions),
                ["enter_short", "enter_tag"]
            ] = (1, "short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场逻辑。
        RL 模型返回的 action 值:
        - 2: Long_exit (做多出场)
        - 4: Short_exit (做空出场)
        """
        # 做多出场: action == 2 (Long_exit)
        exit_long_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 2,
        ]
        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_long_conditions),
                "exit_long"
            ] = 1

        # 做空出场: action == 4 (Short_exit)
        exit_short_conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-action"] == 4,
        ]
        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, exit_short_conditions),
                "exit_short"
            ] = 1

        return dataframe

