import os
import json
import pandas as pd
import numpy as np
import datetime
from scipy.optimize import minimize

import rqdatac
import rqalpha
from rqalpha.api import *

import data_fetcher
import factors

# ----------------- 风险预算求解器 (Risk Budget Solver) -----------------

def solve_risk_budget(cov_matrix, budgets):
    """
    通过数值优化求解风险预算组合权重。
    目标：使各策略对组合跟踪误差的贡献比例等于其信息比的平方比率。
    """
    n = len(budgets)
    init_w = np.array([1.0 / n] * n) # 初始权重等权
    
    # 目标函数：最小化实际风险贡献占比与预算占比的方差和
    def objective(w):
        w = np.array(w)
        portfolio_variance = np.dot(w.T, np.dot(cov_matrix, w))
        portfolio_vol = np.sqrt(portfolio_variance)
        if portfolio_vol == 0:
            return 0
        # 边际风险贡献
        marginal_contrib = np.dot(cov_matrix, w) / portfolio_vol
        # 实际风险贡献
        risk_contrib = w * marginal_contrib / portfolio_vol
        
        # 损失：差异平方和
        loss = np.sum((risk_contrib - budgets) ** 2)
        return loss
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    else:
        # 优化失败则返回默认等权配置
        return init_w

# ----------------- 策略核心逻辑 (RQAlpha 规范) -----------------

def init(context):
    """
    RQAlpha 初始化策略函数
    """
    # 设定中证500指数作为基准
    set_benchmark("000905.XSHG")
    
    # 加载配置参数
    config_path = "/Users/wanglei/Desktop/guosen-backtest/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    context.stock_pool_limit = cfg["stock_pool_limit"]
    context.final_portfolio_limit = cfg["final_portfolio_limit"]
    context.benchmark = cfg["benchmark"]
    
    # 初始化子策略持仓记录与历史绩效
    context.portfolio_A = []  # 策略 A (多因子指数增强)
    context.portfolio_B = []  # 策略 B (超预期精选组合)
    
    # 用于记录两个子组合历史每月超额收益，计算信息比和协方差
    context.excess_history = pd.DataFrame(columns=['excess_A', 'excess_B'])
    context.last_rebalance_date = None
    
    # 注册调度任务：在每个月的最后一个交易日闭市前进行调仓
    scheduler.run_monthly(rebalance, tradingday=-1)
    print("初始化完成：策略复合与风险预算框架设置完毕。")

def rebalance(context, bar):
    """
    月末最后一个交易日触发的策略复合与资产配置调仓主函数
    """
    query_date = context.now.strftime('%Y-%m-%d')
    print(f"\n==================== 调仓日: {query_date} ====================")
    
    # 0. 如果有上月持仓，计算并记录上月的子策略超额收益
    if context.last_rebalance_date is not None and context.portfolio_A and context.portfolio_B:
        record_last_month_performance(context, query_date)
    
    # 1. 动态获取中证500成分股
    try:
        current_components = rqdatac.index_components(context.benchmark, query_date)
    except Exception as e:
        print(f"获取成分股失败: {e}")
        return
        
    if not current_components:
        print("未获取到中证500成分股，跳过本次调仓。")
        return
        
    # 2. 获取基本面财报快照和评级数据 (两策略共用数据源)
    df_snapshot = data_fetcher.get_fundamentals_snapshot(current_components, query_date)
    dt_query = pd.to_datetime(query_date)
    start_rating_date = (dt_query - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
    df_rating = data_fetcher.get_analyst_ratings_count(current_components, start_rating_date, query_date)
    
    # 3. 运行子策略 A 选股 (多因子指数增强 - 月频选前 100 只)
    portfolio_A = factors.calculate_multi_factor_portfolio(
        df_snapshot, 
        current_components, 
        limit=100
    )
    context.portfolio_A = portfolio_A
    
    # 4. 运行子策略 B 选股 (超预期精选组合 - 季频更新持仓，若不是季报月则维持前持仓)
    # 调仓期：每年 1, 4, 7, 8, 10 月末
    current_month = dt_query.month
    if current_month in [1, 4, 7, 8, 10] or not context.portfolio_B:
        print("进入季报换仓期，更新策略 B (超预期精选) 持仓...")
        # 基本面初筛前 60 只
        df_fundamental_rank = factors.calculate_fundamental_factors(
            df_snapshot, 
            df_rating, 
            current_components
        )
        if df_fundamental_rank.empty:
            portfolio_B = current_components[:context.final_portfolio_limit]
        else:
            top_fundamental_stocks = df_fundamental_rank.head(context.stock_pool_limit)['stock_code'].tolist()
            # 用 L2 行情技术打分，精选前 30 只
            fetch_start_date = (dt_query - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            df_daily = data_fetcher.get_daily_行情(top_fundamental_stocks, fetch_start_date, query_date)
            df_benchmark_daily = data_fetcher.get_benchmark_行情(context.benchmark, fetch_start_date, query_date)
            
            if df_daily.empty or df_benchmark_daily.empty:
                portfolio_B = top_fundamental_stocks[:context.final_portfolio_limit]
            else:
                df_final_rank = factors.calculate_technical_factors(
                    top_fundamental_stocks,
                    df_daily,
                    df_benchmark_daily,
                    query_date
                )
                portfolio_B = df_final_rank.head(context.final_portfolio_limit)['stock_code'].tolist()
        context.portfolio_B = portfolio_B
    else:
        print("非季报换仓期，策略 B (超预期精选) 维持原持仓。")
        portfolio_B = context.portfolio_B
        
    # 5. 资产配置：根据信息比率的平方分配风险预算
    w_A, w_B = 0.79, 0.21 # 默认权重 (冷启动前 24 个月)
    
    if len(context.excess_history) >= 24:
        # 使用滚动过去 2 年的历史超额数据计算
        df_roll = context.excess_history.tail(24)
        
        # 计算两策略月度超额收益率的均值与标准差以计算信息比
        mean_A, std_A = df_roll['excess_A'].mean(), df_roll['excess_A'].std()
        mean_B, std_B = df_roll['excess_B'].mean(), df_roll['excess_B'].std()
        
        # 计算滚动信息比
        ir_A = (mean_A / std_A) * np.sqrt(12) if std_A > 0 else 0.0
        ir_B = (mean_B / std_B) * np.sqrt(12) if std_B > 0 else 0.0
        
        # 风险预算比例正比于信息比的平方
        ir_sum_sq = (ir_A ** 2) + (ir_B ** 2)
        if ir_sum_sq > 0:
            budget_ratio = np.array([ir_A ** 2, ir_B ** 2]) / ir_sum_sq
            
            # 计算超额收益协方差矩阵
            cov_matrix = df_roll[['excess_A', 'excess_B']].cov().values
            
            # 求解风险预算权重
            w = solve_risk_budget(cov_matrix, budget_ratio)
            w_A, w_B = w[0], w[1]
            print(f"风险预算配置求解成功 -> 滚动信息比: IR_A={ir_A:.2f}, IR_B={ir_B:.2f}；求解权重: 策略A(指数增强)权重={w_A*100:.2f}%, 策略B(超预期精选)权重={w_B*100:.2f}%")
        else:
            print("计算信息比为非正值，使用默认均权配比。")
    else:
        print(f"历史月度数据不足 24 个月(当前 {len(context.excess_history)} 个月)，使用预设固定权重进行复合：策略A=79.00%, 策略B=21.00%")
        
    # 6. 持仓融合与目标下达
    # 整合 A 和 B 的等权持仓，并根据 w_A 和 w_B 进行融合
    target_weights = {}
    
    # 策略 A 等权
    weight_each_A = 1.0 / len(portfolio_A) if portfolio_A else 0.0
    for stock in portfolio_A:
        target_weights[stock] = target_weights.get(stock, 0.0) + w_A * weight_each_A
        
    # 策略 B 等权
    weight_each_B = 1.0 / len(portfolio_B) if portfolio_B else 0.0
    for stock in portfolio_B:
        target_weights[stock] = target_weights.get(stock, 0.0) + w_B * weight_each_B
        
    # 执行清仓与调仓交易
    current_holdings = list(context.portfolio.positions.keys())
    # A. 卖出不在新目标组合里的股票
    for stock in current_holdings:
        if stock not in target_weights:
            order_target_percent(stock, 0.0)
            
    # B. 对目标持股分配融合后的权重
    for stock, weight in target_weights.items():
        order_target_percent(stock, weight)
        
    # 记录当前调仓状态供下个月统计
    context.last_rebalance_date = query_date
    print("下达多策略融合调仓指令完成。")

def record_last_month_performance(context, query_date):
    """
    计算并记录上月子策略相对于基准指数的超额收益率
    """
    last_date = context.last_rebalance_date
    stocks_A = context.portfolio_A
    stocks_B = context.portfolio_B
    
    # 查询两调仓日之间个股与基准的价格
    try:
        # 基准收益率
        df_bench = rqdatac.get_price(context.benchmark, start_date=last_date, end_date=query_date, frequency='1d', fields=['close'])
        bench_ret = (df_bench.iloc[-1]['close'] / df_bench.iloc[0]['close'] - 1.0) if len(df_bench) >= 2 else 0.0
        
        # 策略 A 股票收益率均值
        df_price_A = rqdatac.get_price(stocks_A, start_date=last_date, end_date=query_date, frequency='1d', fields=['close'])
        rets_A = []
        for stock in stocks_A:
            sub = df_price_A.xs(stock, level='order_book_id') if isinstance(df_price_A.index, pd.MultiIndex) else df_price_A
            if not sub.empty and len(sub) >= 2:
                rets_A.append(sub.iloc[-1]['close'] / sub.iloc[0]['close'] - 1.0)
        ret_A = np.mean(rets_A) if rets_A else 0.0
        
        # 策略 B 股票收益率均值
        df_price_B = rqdatac.get_price(stocks_B, start_date=last_date, end_date=query_date, frequency='1d', fields=['close'])
        rets_B = []
        for stock in stocks_B:
            sub = df_price_B.xs(stock, level='order_book_id') if isinstance(df_price_B.index, pd.MultiIndex) else df_price_B
            if not sub.empty and len(sub) >= 2:
                rets_B.append(sub.iloc[-1]['close'] / sub.iloc[0]['close'] - 1.0)
        ret_B = np.mean(rets_B) if rets_B else 0.0
        
        # 记录超额收益并追加到历史 DataFrame
        excess_A = ret_A - bench_ret
        excess_B = ret_B - bench_ret
        
        new_row = pd.DataFrame([{'excess_A': excess_A, 'excess_B': excess_B}])
        context.excess_history = pd.concat([context.excess_history, new_row], ignore_index=True)
        print(f"上月业绩回溯 -> 策略A超额: {excess_A*100:.2f}%, 策略B超额: {excess_B*100:.2f}% (记录条数: {len(context.excess_history)})")
    except Exception as e:
        print(f"上月业绩回溯计算出错: {e}")

# ----------------- 启动程序与回测结果分析 -----------------

def main():
    # 读取配置参数
    config_path = "config.json"
    if not os.path.exists(config_path):
        config_path = "/Users/wanglei/Desktop/guosen-backtest/config.json"
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    print("--- 启动国信金工中证500增强策略 (基于风险预算多策略复合) ---")
    
    # 初始化 rqdatac
    data_fetcher.init_rq()
    
    # 构造 RQAlpha 回测配置
    rq_config = {
        "base": {
            "start_date": cfg["start_date"],
            "end_date": cfg["end_date"],
            "benchmark": cfg["benchmark"],
            "accounts": {
                "stock": 10000000.0  # 初始资金1000万元
            }
        },
        "extra": {
            "log_level": "error"
        },
        "mod": {
            "sys_analyser": {
                "enabled": True,
                "output_file": "backtest_out.pkl"
            }
        }
    }
    
    # 启动 RQAlpha 回测
    try:
        results = rqalpha.run_func(init=init, config=rq_config)
        
        # 提取并分析回测结果
        analyser = results.get("sys_analyser", {})
        summary = analyser.get("summary", {})
        portfolio = analyser.get("portfolio", pd.DataFrame())
        
        print("\n================== RQAlpha 绩效分析报告 ==================")
        print(f"回测期间: {cfg['start_date']} 至 {cfg['end_date']}")
        print(f"累计收益率: {summary.get('total_returns', 0.0)*100:.2f}%")
        print(f"年化收益率: {summary.get('annualized_returns', 0.0)*100:.2f}%")
        print(f"基准累计收益率: {summary.get('benchmark_total_returns', 0.0)*100:.2f}%")
        print(f"基准年化收益率: {summary.get('benchmark_annualized_returns', 0.0)*100:.2f}%")
        print(f"最大回撤: {summary.get('max_drawdown', 0.0)*100:.2f}%")
        print(f"夏普比率 (Sharpe): {summary.get('sharpe', 0.0):.2f}")
        print(f"信息比率 (Information Ratio): {summary.get('information_ratio', 0.0):.2f}")
        print(f"跟踪误差 (Tracking Error): {summary.get('tracking_error', 0.0)*100:.2f}%")
        print("==========================================================")
        
        # 绘制净值曲线并保存至桌面
        if not portfolio.empty:
            draw_and_save_chart(portfolio)
            
    except Exception as e:
        print(f"回测运行失败: {e}")

def draw_and_save_chart(portfolio):
    """
    绘制净值曲线图
    """
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 7))
        
        dates = portfolio.index
        strat_nav = portfolio['total_value'] / portfolio['total_value'].iloc[0]
        bench_nav = portfolio['benchmark_total_value'] / portfolio['benchmark_total_value'].iloc[0]
        excess_nav = strat_nav / bench_nav
        
        plt.plot(dates, strat_nav, label='复合多策略增强 (RQAlpha)', color='#ca1d18', linewidth=2)
        plt.plot(dates, bench_nav, label='中证500指数基准', color='#666666', linestyle='--', linewidth=1.5)
        plt.plot(dates, excess_nav, label='超额收益净值', color='#1f77b4', linestyle=':', linewidth=1.5)
        
        plt.title('基于风险预算的中证500增强复合策略业绩曲线', fontsize=14)
        plt.xlabel('日期', fontsize=11)
        plt.ylabel('累计净值', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=10)
        
        output_image_path = "/Users/wanglei/Desktop/guosen-backtest/backtest_result.png"
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"\n业绩净值曲线已保存至: {output_image_path}")
        
        # 复制到桌面根目录
        os.system(f"cp {output_image_path} /Users/wanglei/Desktop/策略回测净值业绩图.png")
        print("复制版本已存放至您的桌面根目录：策略回测净值业绩图.png")
        
    except Exception as e:
        print(f"保存可视化图表失败: {e}")

if __name__ == '__main__':
    main()
