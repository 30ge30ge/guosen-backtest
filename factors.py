import pandas as pd
import numpy as np

# ----------------- 子策略 A: 多因子指数增强 -----------------

def calculate_multi_factor_portfolio(df_snapshot, stock_pool, limit=100):
    """
    子策略 A：传统的指数增强多因子打分选股。
    使用估值 (BP近似)、成长 (yoy_net_profit)、盈利 (roe) 维度进行等权打分。
    """
    if df_snapshot.empty:
        return stock_pool[:limit]
        
    df = df_snapshot[df_snapshot['stock_code'].isin(stock_pool)].copy()
    
    # 因子1: 盈利 (roe)
    df['f_roe'] = df['roe'].fillna(df['roe'].median())
    
    # 因子2: 成长 (yoy_net_profit)
    df['f_growth'] = df['yoy_net_profit'].fillna(df['yoy_net_profit'].median())
    
    # 因子3: 估值 (净资产/总市值 近似 BP)
    # total_value 是单位为元的总市值，equity_parent_company_owners 为归母权益(元)
    # BP = 净资产 / 总市值
    df['bp'] = df['equity_parent_company_owners'] / df['total_value']
    df['f_val'] = df['bp'].fillna(df['bp'].median())
    
    # 截面标准化
    for col in ['f_roe', 'f_growth', 'f_val']:
        std_val = df[col].std()
        mean_val = df[col].mean()
        df[col + '_z'] = (df[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        
    # 等权复合得分
    df['factor_score'] = df['f_roe_z'] + df['f_growth_z'] + df['f_val_z']
    
    # 按得分从大到小排序，选择前 N 只
    df_sorted = df.sort_values(by='factor_score', ascending=False)
    return df_sorted.head(limit)['stock_code'].tolist()

# ----------------- 子策略 B: 超预期精选组合 -----------------

def calculate_fundamental_factors(df_snapshot, df_rating_change, stock_pool):
    """
    子策略 B - 步骤1: 基本面初筛。
    剔除单季净利润增速 <= 0 的样本后，通过 ROE、SUE (用 yoy 代替) 等权打分，筛选前 60 只股票池。
    """
    if df_snapshot.empty:
        return pd.DataFrame(columns=['stock_code', 'fundamental_score', 'keep_flag'])
        
    df = df_snapshot[df_snapshot['stock_code'].isin(stock_pool)].copy()
    
    # 1. 过滤：单季度归母净利润同比增速低于 0 的样本剔除
    df['keep_flag'] = df['yoy_net_profit'] > 0
    
    # 2. 因子1: 净资产收益率 roe
    df['roe_score'] = df['roe'].fillna(df['roe'].median())
    
    # 3. 因子2: SUE (此处用 yoy_net_profit 作为惊喜代理)
    df['sue_score'] = df['yoy_net_profit'].fillna(df['yoy_net_profit'].median())
    
    # 4. 因子3: 分析师情绪 (UD_PCT)
    if not df_rating_change.empty:
        up_counts = df_rating_change[df_rating_change['rating_change'] == 'UP'].groupby('order_book_id').size()
        down_counts = df_rating_change[df_rating_change['rating_change'] == 'DOWN'].groupby('order_book_id').size()
        
        rating_df = pd.DataFrame({'up': up_counts, 'down': down_counts}).fillna(0)
        rating_df['ud_pct'] = (rating_df['up'] - rating_df['down']) / (rating_df['up'] + rating_df['down'] + 0.001)
        
        df = df.merge(rating_df['ud_pct'], left_on='stock_code', right_index=True, how='left')
        df['ud_pct'] = df['ud_pct'].fillna(0.0)
    else:
        df['ud_pct'] = df['yoy_net_profit'].fillna(0.0)
        
    # 对三个因子进行 Z-Score
    for col in ['roe_score', 'sue_score', 'ud_pct']:
        std_val = df[col].std()
        mean_val = df[col].mean()
        df[col + '_z'] = (df[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        
    df['fundamental_score'] = df['roe_score_z'] + df['sue_score_z'] + df['ud_pct_z']
    df_valid = df[df['keep_flag'] == True].sort_values(by='fundamental_score', ascending=False)
    
    return df_valid[['stock_code', 'fundamental_score']]

def calculate_technical_factors(stocks, df_daily, df_benchmark, query_date):
    """
    子策略 B - 步骤2: 技术面精选。
    在初筛的 60 只股票中，基于 5 个技术类因子等权打分。
    """
    dt_query = pd.to_datetime(query_date)
    start_lookback = (dt_query - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    
    df_sub = df_daily[(df_daily['date'] >= start_lookback) & (df_daily['date'] <= query_date)].copy()
    df_sub = df_sub[df_sub['order_book_id'].isin(stocks)]
    
    results = []
    
    # 提取基准价格
    bench_sub = df_benchmark[(df_benchmark['date'] >= start_lookback) & (df_benchmark['date'] <= query_date)].sort_values(by='date')
    bench_ret_3d = (bench_sub.iloc[-1]['close'] / bench_sub.iloc[-4]['close'] - 1) if len(bench_sub) >= 4 else 0.0
    
    for stock in stocks:
        stock_df = df_sub[df_sub['order_book_id'] == stock].sort_values(by='date')
        if stock_df.empty or len(stock_df) < 20:
            continue
            
        current_price = stock_df.iloc[-1]['close']
        
        # 1. 52周最高价距离
        max_52w = stock_df['high'].max()
        factor_52w = current_price / max_52w if max_52w > 0 else 0.0
        
        # 2. 非预期换手 (过去20天成交均值 / 过去120天成交均值)
        vol_20 = stock_df.iloc[-20:]['volume'].mean()
        vol_120 = stock_df.iloc[-120:]['volume'].mean()
        factor_turnover = vol_20 / vol_120 if vol_120 > 0 else 1.0
        
        # 3. 规模因子 (市值对数的负向，在打分时取反)
        market_val = stock_df.iloc[-1]['total_value']
        factor_size = np.log(market_val) if market_val > 0 else 0.0
        
        # 4. 公告次日开盘跳空超额 (AOG) 近似计算
        stock_df['ret_open'] = (stock_df['open'] / stock_df['close'].shift(1)) - 1
        factor_aog = stock_df.iloc[-3:]['ret_open'].mean()
        
        # 5. 公告后 3 日超额的近似
        stock_ret_3d = (stock_df.iloc[-1]['close'] / stock_df.iloc[-4]['close'] - 1) if len(stock_df) >= 4 else 0.0
        factor_ret_3d = stock_ret_3d - bench_ret_3d
        
        results.append({
            'stock_code': stock,
            'f_52w': factor_52w,
            'f_turnover': factor_turnover,
            'f_size': factor_size,
            'f_aog': factor_aog,
            'f_ret_3d': factor_ret_3d
        })
        
    df_tech = pd.DataFrame(results)
    if df_tech.empty:
        return pd.DataFrame(columns=['stock_code', 'tech_score'])
        
    # 规模反向打分
    df_tech['f_size'] = -df_tech['f_size']
    
    for col in ['f_52w', 'f_turnover', 'f_size', 'f_aog', 'f_ret_3d']:
        std_val = df_tech[col].std()
        mean_val = df_tech[col].mean()
        df_tech[col + '_z'] = (df_tech[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        
    df_tech['tech_score'] = df_tech['f_52w_z'] + df_tech['f_turnover_z'] + df_tech['f_size_z'] + df_tech['f_aog_z'] + df_tech['f_ret_3d_z']
    
    return df_tech.sort_values(by='tech_score', ascending=False)
