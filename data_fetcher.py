import pandas as pd
import numpy as np
import datetime
import rqdatac

# 初始化米筐 API
def init_rq():
    """
    初始化 rqdatac。支持本地全局认证，即使不传入凭证也会自动加载本地配置。
    """
    try:
        rqdatac.init()
        print("RQData 初始化成功！")
    except Exception as e:
        print(f"RQData 初始化失败，请检查本地认证配置: {e}")

def get_trading_days(start_date, end_date):
    """
    获取交易日历
    """
    dates = rqdatac.get_trading_dates(start_date, end_date)
    return [d.strftime('%Y-%m-%d') for d in dates]

def get_index_weights_or_components(index_code, date):
    """
    获取指数成分股
    """
    try:
        # 获取指数成分股列表
        components = rqdatac.index_components(index_code, date)
        return components
    except Exception as e:
        print(f"获取指数成分股失败 (date={date}): {e}")
        return []

def get_daily_行情(stocks, start_date, end_date):
    """
    批量获取个股的日频行情数据，包含 open, high, low, close, volume, limit_status(涨跌停状态)
    """
    if not stocks:
        return pd.DataFrame()
    try:
        df = rqdatac.get_price(
            stocks, 
            start_date=start_date, 
            end_date=end_date, 
            frequency='1d', 
            fields=['open', 'high', 'low', 'close', 'volume', 'limit_status', 'total_value']
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 重置索引方便 Pandas 处理
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        print(f"获取个股行情失败: {e}")
        return pd.DataFrame()

def get_benchmark_行情(benchmark, start_date, end_date):
    """
    获取基准指数的日频行情
    """
    try:
        df = rqdatac.get_price(
            benchmark, 
            start_date=start_date, 
            end_date=end_date, 
            frequency='1d', 
            fields=['open', 'high', 'low', 'close', 'volume']
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        print(f"获取基准指数行情失败: {e}")
        return pd.DataFrame()

def get_financial_reports(stocks, start_date, end_date):
    """
    获取个股的季度财务报表指标，用于计算基本面因子。
    使用 get_fundamentals 获取，并包含报告期与公布日期 pub_date 以防止未来函数。
    """
    if not stocks:
        return pd.DataFrame()
    
    # 构造查询，获取归母净利润、营业收入、净资产、ROE等字段
    # 根据米筐定义：
    # income_statement.np_parent_company_owners: 归属于母公司所有者的净利润
    # income_statement.revenue: 营业总收入
    # balance_sheet.equity_parent_company_owners: 归属于母公司所有者权益合计
    # financial_indicator.roe: 净资产收益率 (TTM 或季度)
    # financial_indicator.yoy_net_profit: 净利润同比增长率 (YOY)
    
    try:
        # 批量获取指定时间区间内的季度财报数据
        # 采用 get_fundamentals_quarterly (季度财务数据，方便计算季度环比/同比和标准化预期外盈利 SUE)
        q = rqdatac.query(
            rqdatac.fundamentals.income_statement.np_parent_company_owners,
            rqdatac.fundamentals.income_statement.revenue,
            rqdatac.fundamentals.balance_sheet.equity_parent_company_owners,
            rqdatac.fundamentals.financial_indicator.roe,
            rqdatac.fundamentals.financial_indicator.yoy_net_profit
        ).filter(
            rqdatac.fundamentals.income_statement.stock_code.in_(stocks)
        )
        
        # 每次获取一个年度/季度的财务指标，或者通过滚动获取。
        # 为了高效起见，我们采用 get_fundamentals 查询指定日期列表，或者直接获取历史财务序列。
        # 米筐的 get_fundamentals(query, date) 获取特定日期已公布的最新的财务报表
        # 为了能够在回测中防止未来函数，我们需要在“调仓日”去查询当时已公布的最新财报。
        # 这里我们在外层循环的每个调仓日调用此函数，获取该日已披露的财务快照。
        return q
    except Exception as e:
        print(f"获取财务数据失败: {e}")
        return None

def get_fundamentals_snapshot(stocks, query_date):
    """
    获取指定日期已披露的最新的基本面截面快照
    """
    if not stocks:
        return pd.DataFrame()
    try:
        # 查询主要的基本面因子
        df = rqdatac.get_fundamentals(
            rqdatac.query(
                rqdatac.fundamentals.income_statement.np_parent_company_owners, # 归母净利润
                rqdatac.fundamentals.income_statement.revenue,                  # 营业收入
                rqdatac.fundamentals.balance_sheet.equity_parent_company_owners, # 归母权益
                rqdatac.fundamentals.financial_indicator.roe,                   # 净资产收益率
                rqdatac.fundamentals.financial_indicator.yoy_net_profit,        # 净利润同比增长率
                # 附带公布日期，避免未来函数
                rqdatac.fundamentals.income_statement.pub_date,
                rqdatac.fundamentals.income_statement.report_date
            ).filter(
                rqdatac.fundamentals.income_statement.stock_code.in_(stocks)
            ),
            entry_date=query_date,
            interval='1d'
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 处理多重索引
        df = df.xs(query_date, level='date')
        df = df.reset_index()
        df = df.rename(columns={'order_book_id': 'stock_code'})
        return df
    except Exception as e:
        # 兼容处理
        print(f"获取截面财务快照失败 (date={query_date}): {e}")
        return pd.DataFrame()

def get_analyst_ratings_count(stocks, start_date, end_date):
    """
    获取过去3个月分析师关于个股评级上调/下调的数据。
    米筐中有 get_share_rating 接口返回个股的分析师评级情况。
    """
    if not stocks:
        return pd.DataFrame()
    try:
        # 获取评级变动历史。字段包括 rating_change: 'UP' (上调), 'DOWN' (下调), 'KEEP' (维持), 'NEW' (新增)
        # 接口：rqdatac.get_share_rating(order_book_ids, start_date, end_date)
        df = rqdatac.get_share_rating(stocks, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        return df
    except Exception as e:
        print(f"获取分析师评级变动失败: {e} (可能无权限，将使用备用景气度因子)")
        return pd.DataFrame()
