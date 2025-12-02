import logging

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
    stoploss = -0.05
    timeframe = "5m"

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
        return dataframe

    # --------------------------------------
    # 2. 目标定义 (Target Definition)
    # --------------------------------------

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        定义模型训练的目标(Labels)。
        对于 RL(强化学习)模型, 这些目标通常用于计算 Reward 或辅助分析。
        注意: 所有目标列名必须以 '&' 开头。
        """
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )
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
        """
        # 示例: 基于 do_predict 标志和模型预测结果(如果有显式预测值)
        # 对于 RL 模型, 通常由模型直接给出 Action, 这里可以是简单的占位符,
        # 或者根据 FreqAI 返回的特定列进行判断。

        # 这里是一个通用的 FreqAI 进场模板:
        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            # 如果是回归/分类模型, 可能会判断预测值:
            # dataframe["&s-close"] > 0.01
        ]

        if enter_long_conditions:
             dataframe.loc[
                (enter_long_conditions[0]), # 简化逻辑
                'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        出场逻辑。
        """
        return dataframe

