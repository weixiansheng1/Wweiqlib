# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 20:44:29 2025

@author: 11298
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TopkDropoutStrategy 和 SimulatorExecutor 源码分析演示

本示例展示如何使用 TopkDropoutStrategy 和 SimulatorExecutor 进行量化投资回测，
并深入解析其内部工作机制。

"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


import qlib
from qlib.config import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest.account import Account
from qlib.backtest.exchange import Exchange
from qlib.data.dataset import DatasetH
from qlib.workflow import R
from qlib.backtest import backtest
from qlib.utils import init_instance_by_config
from qlib.model.trainer import TrainerR

import torch



def initialize_qlib():
    
    """初始化Qlib环境"""
    #验证CPU版本
    print("PyTorch版本:", torch.__version__)
    print("CPU支持:", torch.cpu.is_available())

    #验证GPU版本（如有）
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")

    print("🚀 初始化Qlib环境...")
    try:
        # 初始化Qlib，使用本地数据
        provider_uri = "~/.qlib/qlib_data/cn_data"  # 数据路径
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        print("✅ Qlib初始化成功")
        return True
    except Exception as e:
        print(f"❌ Qlib初始化失败: {e}")
        return False

def create_dataset():
    """创建数据集"""
    print("\n📊 创建Alpha158特征数据集...")
    
    # Alpha158数据处理器配置
    handler_config = {
        "start_time": "2020-01-01",
        "end_time": "2020-12-31", 
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2020-08-31",
        "instruments": "csi300",  # 沪深300成分股
        "infer_processors": [
            {
                "class": "RobustZScoreNorm",
                "kwargs": {"fields_group": "feature", "clip_outlier": True},
            },
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],  # 未来2日收益率
    }
    
    # 创建Alpha158处理器
    handler = Alpha158(**handler_config)
    
    # 定义数据集分段
    segments = {
        "train": ("2020-01-01", "2020-06-30"),
        "valid": ("2020-07-01", "2020-08-31"), 
        "test": ("2020-09-01", "2020-12-31")
    }
    
    # 创建数据集
    dataset = DatasetH(handler, segments=segments)
    
    print("✅ 数据集创建成功")
    print(f"   - 特征维度: Alpha158 (158个技术指标)")
    print(f"   - 标的池: CSI300成分股") 
    print(f"   - 时间范围: 2020-01-01 到 2020-12-31")
    print(f"   - 标签: 未来2日收益率")
    
    return dataset

def train_model(dataset):
    """训练LightGBM预测模型"""
    print("\n🤖 训练LightGBM预测模型...")
    
    # 直接创建LGBModel实例
    model = LGBModel(
        objective="regression",
        num_leaves=60,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        num_boost_round=100,
        early_stopping_rounds=10,
    )
    
    # 训练模型
    model.fit(dataset)
    
    print("✅ 模型训练完成")
    print("   - 模型类型: LightGBM")
    print("   - 目标函数: 回归")
    print("   - 叶子数量: 60")
    print("   - 学习率: 0.1")
    
    return model

def analyze_topk_dropout_strategy():
    """分析TopkDropoutStrategy的核心机制"""
    print("\n🔍 TopkDropoutStrategy 源码机制分析")
    print("=" * 60)
    
    # 1. 核心参数解析
    print("📋 核心参数说明:")
    print("   • topk: 持仓股票数量 (如: 50只)")
    print("   • n_drop: 每日调仓数量 (如: 5只)")
    print("   • method_sell: 卖出策略")
    print("     - 'bottom': 卖出得分最低的股票")
    print("     - 'random': 随机卖出")
    print("   • method_buy: 买入策略") 
    print("     - 'top': 买入得分最高的未持有股票")
    print("     - 'random': 随机买入")
    print("   • hold_thresh: 最小持有天数")
    print("   • only_tradable: 是否只考虑可交易股票")
    print("   • forbid_all_trade_at_limit: 涨跌停时是否禁止交易")
    
    # 2. 算法流程分析
    print("\n🔄 核心算法流程:")
    print("   1️⃣ 获取当前交易日的预测信号")
    print("   2️⃣ 分析当前持仓，按得分排序")  
    print("   3️⃣ 选择买入候选股票 (未持有的高得分股票)")
    print("   4️⃣ 确定卖出股票 (避免卖高买低)")
    print("   5️⃣ 生成具体的买卖订单")
    
    # 3. 关键设计亮点
    print("\n💡 关键设计亮点:")
    print("   ✨ 防卖高买低: 通过comb机制确保交易合理性")
    print("   ✨ 持有时间控制: hold_thresh避免过度频繁交易")
    print("   ✨ 可交易性检查: 考虑股票实际可交易状态")
    print("   ✨ 风险管理: risk_degree控制总体仓位")

def analyze_simulator_executor():
    """分析SimulatorExecutor的核心机制"""
    print("\n🎮 SimulatorExecutor 执行器机制分析")
    print("=" * 60)
    
    # 1. 执行模式分析
    print("⚡ 交易执行模式:")
    print("   • TT_SERIAL (串行模式):")
    print("     - 订单按序执行")
    print("     - 支持先卖后买 (释放资金再投资)")
    print("     - 适合日频交易")
    print("   • TT_PARAL (并行模式):")
    print("     - 订单并行执行") 
    print("     - 卖单优先执行 (避免资金冲突)")
    print("     - 适合高频交易场景")
    
    # 2. 核心功能
    print("\n🛠️ 核心功能模块:")
    print("   📦 订单管理: _get_order_iterator() 处理订单序列")
    print("   💰 资金管理: 实时跟踪账户资金变化")
    print("   📊 交易记录: 详细记录每笔交易的执行情况")
    print("   🔄 日内跟踪: dealt_order_amount 跟踪日内累计交易")
    
    # 3. 集成特点
    print("\n🔗 基础设施集成:")
    print("   🏪 Exchange: 模拟真实交易所环境")
    print("   💳 Account: 管理资金和持仓状态")
    print("   📅 Calendar: 管理交易日历和时间")
    print("   📈 Metrics: 生成交易和组合指标")

def create_strategy_config(model, dataset):
    """创建策略配置"""
    print("\n⚙️ 配置TopkDropoutStrategy...")
    
    # 策略配置
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "kwargs": {
            "signal": (model, dataset),  # 模型和数据集
            "topk": 20,              # 持有20只股票
            "n_drop": 3,             # 每日调仓3只
            "method_sell": "bottom", # 卖出得分最低的股票
            "method_buy": "top",     # 买入得分最高的股票  
            "hold_thresh": 1,        # 最少持有1天
            "only_tradable": True,   # 只考虑可交易股票
            "forbid_all_trade_at_limit": True,  # 涨跌停时禁止交易
        },
    }
    
    print("✅ 策略配置完成")
    print(f"   - 持仓数量: {strategy_config['kwargs']['topk']}只")
    print(f"   - 日调仓量: {strategy_config['kwargs']['n_drop']}只") 
    print(f"   - 换手率: {strategy_config['kwargs']['n_drop']/strategy_config['kwargs']['topk']*100:.1f}%")
    
    return strategy_config

def create_executor_config():
    """创建执行器配置"""
    print("\n⚙️ 配置SimulatorExecutor...")
    
    # 执行器配置
    executor_config = {
        "class": "SimulatorExecutor",
        "kwargs": {
            "time_per_step": "day",           # 日频交易
            "generate_portfolio_metrics": True,  # 生成组合指标
            "verbose": False,                 # 不显示详细交易信息
            "trade_type": SimulatorExecutor.TT_SERIAL,  # 串行执行模式
            "indicator_config": {             # 交易指标配置
                "show_indicator": True,
                "pa_config": {
                    "base_price": "twap",     # 基于时间加权平均价
                    "weight_method": "mean",  # 均值加权
                },
                "ffr_config": {
                    "weight_method": "value_weighted",  # 按价值加权
                }
            }
        },
    }
    
    print("✅ 执行器配置完成")
    print(f"   - 执行频率: {executor_config['kwargs']['time_per_step']}")
    print(f"   - 执行模式: {executor_config['kwargs']['trade_type']}")
    print("   - 指标监控: 启用价格优势和成交率监控")
    
    return executor_config

def run_backtest(strategy_config, executor_config):
    """运行回测"""
    print("\n🚀 运行量化回测...")
    print("=" * 60)
    
    # 回测配置
    backtest_config = {
        "start_time": "2020-09-01",
        "end_time": "2020-12-31", 
        "account": 1000000,       # 初始资金100万
        "benchmark": "SH000300",  # 沪深300基准
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,  # 涨跌停限制
            "deal_price": "close",     # 收盘价成交
            "open_cost": 0.0005,       # 手续费0.05%
            "close_cost": 0.0005,
            "min_cost": 5,             # 最小手续费5元
        },
    }
    
    # 执行回测
    print("📊 开始回测计算...")
    
    try:
        # 运行回测
        portfolio_metric_dict, indicator_dict = backtest(
            executor=executor_config,
            strategy=strategy_config, 
            **backtest_config
        )
        
        print("✅ 回测计算完成")
        
        # 分析回测结果
        analyze_backtest_results(portfolio_metric_dict, indicator_dict)
        
        return portfolio_metric_dict, indicator_dict
        
    except Exception as e:
        print(f"❌ 回测执行失败: {e}")
        return None, None

def analyze_backtest_results(portfolio_metric_dict, indicator_dict):
    """分析回测结果"""
    print("\n📈 回测结果分析")
    print("=" * 60)
    
    if portfolio_metric_dict is None:
        print("❌ 无有效回测结果")
        return
        
    try:
        # 提取关键指标
        excess_return_without_cost = portfolio_metric_dict.get('excess_return_without_cost', {})
        excess_return_with_cost = portfolio_metric_dict.get('excess_return_with_cost', {})
        
        print("📊 投资组合表现:")
        if 'annualized_return' in excess_return_without_cost:
            print(f"   📈 年化收益率 (无成本): {excess_return_without_cost['annualized_return']:.2%}")
        if 'annualized_return' in excess_return_with_cost:  
            print(f"   💰 年化收益率 (含成本): {excess_return_with_cost['annualized_return']:.2%}")
        if 'information_ratio' in excess_return_with_cost:
            print(f"   📏 信息比率: {excess_return_with_cost['information_ratio']:.3f}")
        if 'max_drawdown' in excess_return_with_cost:
            print(f"   📉 最大回撤: {excess_return_with_cost['max_drawdown']:.2%}")
        
        # 交易指标分析
        if indicator_dict:
            print("\n💹 交易执行分析:")
            if '1day.pa' in indicator_dict:
                pa = indicator_dict['1day.pa']
                if isinstance(pa, (int, float)):
                    print(f"   ⚡ 价格优势 (PA): {pa:.4f}")
            if '1day.ffr' in indicator_dict:
                ffr = indicator_dict['1day.ffr'] 
                if isinstance(ffr, (int, float)):
                    print(f"   ✅ 成交率 (FFR): {ffr:.2%}")
                    
    except Exception as e:
        print(f"⚠️ 结果分析出现问题: {e}")
        print("📋 可用指标键名:", list(portfolio_metric_dict.keys()) if portfolio_metric_dict else "无")

def demonstrate_strategy_mechanism():
    """演示策略内部工作机制"""
    print("\n🔬 策略机制深度解析")
    print("=" * 60)
    
    # 模拟策略决策过程
    print("🎯 模拟TopkDropoutStrategy决策过程:")
    
    # 创建模拟数据
    np.random.seed(42)
    stocks = [f"股票{i:03d}" for i in range(100)]
    pred_scores = pd.Series(np.random.randn(100), index=stocks, name='score')
    current_holdings = stocks[:20]  # 当前持有前20只
    
    print(f"\n📊 当前市场状态:")
    print(f"   • 股票池大小: {len(stocks)}只")
    print(f"   • 当前持仓: {len(current_holdings)}只") 
    print(f"   • 预测信号范围: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
    
    # 模拟策略参数
    topk, n_drop = 20, 3
    
    # 步骤1: 当前持仓排序
    current_scores = pred_scores[current_holdings].sort_values(ascending=False)
    print(f"\n🔹 步骤1 - 当前持仓分析:")
    print(f"   最佳持仓: {current_scores.index[0]} (得分: {current_scores.iloc[0]:.3f})")
    print(f"   最差持仓: {current_scores.index[-1]} (得分: {current_scores.iloc[-1]:.3f})")
    
    # 步骤2: 买入候选选择  
    not_holding = pred_scores[~pred_scores.index.isin(current_holdings)]
    buy_candidates = not_holding.sort_values(ascending=False).head(n_drop + topk - len(current_holdings))
    print(f"\n🔹 步骤2 - 买入候选分析:")
    print(f"   候选数量: {len(buy_candidates)}只")
    print(f"   最佳候选: {buy_candidates.index[0]} (得分: {buy_candidates.iloc[0]:.3f})")
    
    # 步骤3: 避免卖高买低
    combined = pd.concat([current_scores, buy_candidates]).sort_values(ascending=False)
    sell_candidates = current_scores[current_scores.index.isin(combined.tail(n_drop).index)]
    print(f"\n🔹 步骤3 - 卖出决策分析:")
    print(f"   待卖数量: {len(sell_candidates)}只")
    if len(sell_candidates) > 0:
        print(f"   将卖出: {sell_candidates.index[0]} (得分: {sell_candidates.iloc[0]:.3f})")
    
    # 步骤4: 最终交易决策
    actual_buy = buy_candidates.head(len(sell_candidates))
    print(f"\n🔹 步骤4 - 最终交易决策:")
    print(f"   卖出股票数: {len(sell_candidates)}")
    print(f"   买入股票数: {len(actual_buy)}")
    print(f"   预期换手率: {len(sell_candidates)/topk*100:.1f}%")

def main():
    """主函数"""
    print("🎯 TopkDropoutStrategy 和 SimulatorExecutor 源码分析演示")
    print("=" * 80)
    
    # 初始化环境
    if not initialize_qlib():
        print("❌ 环境初始化失败，程序退出")
        return
    
    try:
        # 源码机制分析
        analyze_topk_dropout_strategy()
        analyze_simulator_executor() 
        
        # 演示策略内部机制
        demonstrate_strategy_mechanism()
        
        # 创建数据集
        dataset = create_dataset()
        
        # 训练模型
        model = train_model(dataset)
        
        # 配置策略和执行器
        strategy_config = create_strategy_config(model, dataset)
        executor_config = create_executor_config()
        
        # 运行回测
        portfolio_results, indicator_results = run_backtest(strategy_config, executor_config)
        
        print("\n🎉 演示完成!")
        print("=" * 80)
        print("📚 关键知识点总结:")
        print("   1️⃣ TopkDropoutStrategy实现智能选股和动态调仓")
        print("   2️⃣ SimulatorExecutor提供高保真的交易执行模拟")
        print("   3️⃣ 两者协同工作构成完整的量化回测系统") 
        print("   4️⃣ 模块化设计支持灵活的策略定制和优化")
      
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
