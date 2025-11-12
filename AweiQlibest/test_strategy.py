from pprint import pprint
import pandas as pd
import numpy as np
import qlib
from qlib.utils.time import Freq
from qlib.utils import flatten_dict
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.data import D  # 用于获取股票列表和日历


# 1. 初始化Qlib（确保数据路径正确）
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")

# 2. 定义常量
CSI300_BENCH = "SH000300"  # 基准指数
START_TIME = "2017-01-01"
END_TIME = "2020-08-01"
FREQ = "day"


# 3. 生成预测信号 pred_score（关键步骤）
# 说明：实际应用中需替换为模型的预测结果（如LightGBM/XGBoost的输出）
# 此处用随机数据模拟，格式为 pd.Series，索引为 (datetime, instrument) 的 MultiIndex
def generate_pred_score():
    # 获取CSI300成分股作为股票池
    instruments = D.instruments(market="csi300")  # 从数据中获取沪深300成分股
    if not instruments:
        raise ValueError("未获取到股票列表，请检查数据是否正确加载")
    
    # 获取回测时间范围内的交易日历
    cal = D.calendar(start_time=START_TIME, end_time=END_TIME, freq=FREQ)
    if len(cal) == 0:
        raise ValueError("未获取到交易日历，请检查数据是否完整")
    
    # 构造 MultiIndex（datetime, instrument）
    index = pd.MultiIndex.from_product(
        [cal, instruments],
        names=["datetime", "instrument"]
    )
    
    # 生成随机预测分数（替代模型预测，实际应替换为真实预测结果）
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    pred_score = pd.Series(
        np.random.randn(len(index)),  # 随机正态分布分数（越高表示越看好）
        index=index,
        name="pred_score"
    )
    return pred_score


# 生成信号（必须在策略初始化前完成）
pred_score = generate_pred_score()


# 4. 策略配置
STRATEGY_CONFIG = {
    "topk": 50,          # 选取预测分数前50的股票
    "n_drop": 5,         # 每次调仓剔除5只股票（避免过度交易）
    "signal": pred_score,  # 传入生成的预测信号
}


# 5. 初始化策略
strategy_obj = TopkDropoutStrategy(** STRATEGY_CONFIG)


# 6. 执行回测（补充必要参数）
report_normal, positions_normal = backtest_daily(
    start_time=START_TIME,
    end_time=END_TIME,
    strategy=strategy_obj,
    benchmark=CSI300_BENCH,  # 基准指数
    account=100000000,      # 初始资金（1亿）
    # 交易成本配置（与实际市场匹配）
    exchange_kwargs={
        "freq": FREQ,
        "limit_threshold": 0.095,  # 涨跌停限制（9.5%）
        "deal_price": "close",     # 以收盘价成交
        "open_cost": 0.0005,       # 开仓手续费（0.05%）
        "close_cost": 0.0015,      # 平仓手续费（0.15%）
        "min_cost": 5,             # 最低手续费（5元）
    },
)


# 7. 结果分析
analysis = {
    # 无手续费的超额收益（策略收益 - 基准收益）
    "excess_return_without_cost": risk_analysis(
        report_normal["return"] - report_normal["bench"],
        freq=FREQ
    ),
    # 含手续费的超额收益（扣除交易成本后）
    "excess_return_with_cost": risk_analysis(
        report_normal["return"] - report_normal["bench"] - report_normal["cost"],
        freq=FREQ
    ),
}

# 合并分析结果并打印
analysis_df = pd.concat(analysis)
print("回测结果分析：")
pprint(analysis_df)