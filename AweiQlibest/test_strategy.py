
# init qlib

from pprint import pprint
import pandas as pd
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D  # 补充导入
from qlib import init

# 1. 初始化Qlib（确保数据存在）
init(provider_uri="~/.qlib/qlib_data/cn_data")

# 2. 定义常量和配置
CSI300_BENCH = "SH000300"
FREQ = "day"

# 3. 生成预测分数 pred_score（关键步骤，需根据实际模型替换）
# 示例：假设从测试数据中获取（实际需用模型预测）
# 这里仅为演示，实际需替换为你的模型预测结果
def generate_pred_score():
    # 假设选取CSI300成分股作为示例
    instruments = D.instruments(market="csi300")
    # 用随机数模拟预测分数（实际需替换为模型输出）
    dates = pd.date_range(start="2017-01-01", end="2020-08-01", freq="D")
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    pred_score = pd.Series(range(len(index)), index=index, name="score")
    return pred_score

pred_score = generate_pred_score()  # 生成信号

# 4. 策略、执行器、回测配置
STRATEGY_CONFIG = {
    "topk": 50,          # 选取预测分数前50的股票
    "n_drop": 5,         # 每次调仓剔除5只股票
    "signal": pred_score,  # 传入预测分数
}

EXECUTOR_CONFIG = {
    "time_per_step": "day",  # 每日调仓
    "generate_portfolio_metrics": True,  # 生成组合指标
}

backtest_config = {
    "start_time": "2017-01-01",
    "end_time": "2020-08-01",
    "account": 100000000,  # 初始资金
    "benchmark": CSI300_BENCH,  # 基准指数
    "exchange_kwargs": {
        "freq": FREQ,
        "limit_threshold": 0.095,  # 涨跌停限制
        "deal_price": "close",  # 以收盘价成交
        "open_cost": 0.0005,    # 开仓手续费
        "close_cost": 0.0015,   # 平仓手续费
        "min_cost": 5,          # 最低手续费
    },
}

# 5. 执行回测
strategy_obj = TopkDropoutStrategy(** STRATEGY_CONFIG)
executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
portfolio_metric_dict, indicator_dict = backtest(
    executor=executor_obj,
    strategy=strategy_obj,** backtest_config
)

# 6. 结果分析
analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

analysis = {
    "excess_return_without_cost": risk_analysis(
        report_normal["return"] - report_normal["bench"], freq=analysis_freq
    ),
    "excess_return_with_cost": risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq
    ),
}

analysis_df = pd.concat(analysis)
analysis_dict = flatten_dict(analysis_df["risk"].unstack().T.to_dict())

# 打印结果
print(f"基准收益分析 ({analysis_freq}):")
pprint(risk_analysis(report_normal["bench"], freq=analysis_freq))
print(f"\n无手续费超额收益分析 ({analysis_freq}):")
pprint(analysis["excess_return_without_cost"])
print(f"\n含手续费超额收益分析 ({analysis_freq}):")
pprint(analysis["excess_return_with_cost"])