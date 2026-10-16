"""Daily aggregation and feature engineering, identical to Uttam_EDA.ipynb sections 20-21."""
import numpy as np
import pandas as pd
import polars as pl

TECH = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'vol_20', 'rsi_14', 'macd_pct', 'macd_hist_pct',
        'vol_z20', 'turnover_ratio_20', 'dist_high20', 'dist_low20', 'range_fraction',
        'intraday_return', 'open_return', 'vwap_deviation', 'gap_days', 'log_turnover',
        'log_trade_count', 'thin']
MARKET = ['mkt_ret', 'mkt_ret_5', 'breadth', 'mkt_turnover_z', 'excess_ret_1', 'excess_ret_5',
          'excess_sector_ret']
BROKER = ['broker_imbalance', 'imbalance_5', 'buyer_hhi', 'seller_hhi', 'hhi_diff', 'hhi_diff_5',
          'top_buyer_share', 'top_seller_share', 'crossed_share', 'log_buyer_seller_ratio',
          'log_n_buyers']
MIN_HISTORY = 20  # sessions of history a stock needs before it gets a signal


def _scan(csv_path):
    return pl.scan_csv(csv_path, try_parse_dates=True, schema_overrides={'contract_no': pl.Int64})


def aggregate_daily(csv_path):
    """One row per (symbol, date) from the raw floorsheet CSV."""
    return (_scan(csv_path).group_by(['symbol', 'date']).agg(
        pl.col('company_name').first(),
        pl.col('rate').sort_by('contract_no').first().alias('open'),
        pl.col('rate').max().alias('high'),
        pl.col('rate').min().alias('low'),
        pl.col('rate').sort_by('contract_no').last().alias('close'),
        pl.col('quantity').sum().alias('volume'),
        pl.col('amount').sum().alias('turnover'),
        pl.len().alias('trade_count'),
        pl.col('quantity').filter(pl.col('buyer_broker_code') == pl.col('seller_broker_code'))
          .sum().alias('crossed_volume'),
        pl.col('buyer_broker_code').n_unique().alias('n_buyers'),
        pl.col('seller_broker_code').n_unique().alias('n_sellers'))
        .collect().sort(['symbol', 'date']))


def aggregate_brokers(csv_path):
    """Per (symbol, date) broker concentration and net-flow statistics."""
    s = _scan(csv_path)
    buys = (s.group_by(['symbol', 'date', 'buyer_broker_code'])
             .agg(pl.col('quantity').sum().alias('bought')).rename({'buyer_broker_code': 'broker'}))
    sells = (s.group_by(['symbol', 'date', 'seller_broker_code'])
              .agg(pl.col('quantity').sum().alias('sold')).rename({'seller_broker_code': 'broker'}))
    flows = (buys.join(sells, on=['symbol', 'date', 'broker'], how='full', coalesce=True)
        .with_columns(pl.col('bought').fill_null(0), pl.col('sold').fill_null(0))
        .with_columns((pl.col('bought') - pl.col('sold')).alias('net')))
    return (flows.group_by(['symbol', 'date']).agg(
        ((pl.col('bought')/pl.col('bought').sum())**2).sum().alias('buyer_hhi'),
        ((pl.col('sold')/pl.col('sold').sum())**2).sum().alias('seller_hhi'),
        (pl.col('bought').max()/pl.col('bought').sum()).alias('top_buyer_share'),
        (pl.col('sold').max()/pl.col('sold').sum()).alias('top_seller_share'),
        pl.col('net').sort(descending=True).head(5).clip(lower_bound=0).sum().alias('top5_net_buy'),
        pl.col('net').sort().head(5).clip(upper_bound=0).sum().alias('top5_net_sell'))
        .collect().sort(['symbol', 'date']))


def sector_of(name):
    """Keyword heuristic from the company name; not an official NEPSE classification."""
    n = name.lower()
    if '%' in n or any(k in n for k in ('debenture', 'bond', 'rinpatra')):
        return 'Debenture'
    if any(k in n for k in ('fund', 'yojana', 'kosh', 'scheme', 'select-30', 'super 30',
                            '80-20', 'nmb 50', 'focus 40')):
        return 'Mutual fund'
    if 'laghubitta' in n or 'microfinance' in n:
        return 'Microfinance'
    if 'Life' in name and ('Insurance' in name or 'Corporation' in name):
        return 'Life insurance'
    if 'insurance' in n or 'beema' in n:
        return 'Non-life insurance'
    if any(k in n for k in ('bikas bank', 'bikash bank', 'development bank')):
        return 'Development bank'
    if 'finance' in n or 'merchant banking' in n:
        return 'Finance'
    if 'bank' in n:
        return 'Commercial bank'
    if any(k in n for k in ('hydro', 'power', 'energy', 'urja', 'jal', 'bidyut', 'bidhyut')):
        return 'Hydropower'
    if any(k in n for k in ('hotel', 'cable car', 'hills', 'darshan', 'tourism')):
        return 'Hotels & tourism'
    if any(k in n for k in ('investment', 'holdings', 'trust', 'ventures', 'emerging nepal')):
        return 'Investment'
    if any(k in n for k in ('cement', 'bottlers', 'distillery', 'pharmaceutical', 'unilever',
                            'spinning', 'lube', 'colour', 'panel', 'minerals', 'agritech', 'krishi')):
        return 'Manufacturing'
    return 'Others'


def build_features(daily, broker, sectors, horizon=5, k_vol=.5, min_move=.01):
    """Return one pandas row per stock-day with every model input, plus the realised target."""
    p = daily.join(broker, on=['symbol', 'date'], how='left')
    p = pd.DataFrame(p.to_dict(as_series=False))
    p['date'] = pd.to_datetime(p['date'])
    p = p.sort_values(['symbol', 'date'], kind='stable').reset_index(drop=True)
    p['sector'] = pd.Categorical(p.company_name.map(sector_of), categories=sectors)
    p['promoter'] = p.company_name.str.contains('promot', case=False).astype(float)

    g = p.groupby('symbol', sort=False)
    c, prev = p['close'], g['close'].shift(1)
    p['ret_1'] = c/prev - 1
    for k in (5, 10, 20):
        p[f'ret_{k}'] = c/g['close'].shift(k) - 1
    roll = lambda col, fn, w=20, m=10: g[col].transform(lambda s: getattr(s.rolling(w, min_periods=m), fn)())
    p['vol_20'] = roll('ret_1', 'std')
    diff = g['close'].diff()
    p['_up'], p['_down'] = diff.clip(lower=0), -diff.clip(upper=0)
    wilder = lambda col: g[col].transform(lambda s: s.ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    up, down = wilder('_up'), wilder('_down')
    p['rsi_14'] = np.where(down > 0, 100 - 100/(1 + up/down.where(down > 0)), 100.)
    p.loc[up.isna(), 'rsi_14'] = np.nan
    ema = lambda col, span: g[col].transform(lambda s: s.ewm(span=span, adjust=False).mean())
    p['_macd'] = ema('close', 12) - ema('close', 26)
    p['macd_pct'] = p['_macd']/c
    p['macd_hist_pct'] = (p['_macd'] - ema('_macd', 9))/c
    p['log_volume'] = np.log1p(p.volume)
    p['vol_z20'] = (p.log_volume - roll('log_volume', 'mean'))/roll('log_volume', 'std')
    p['turnover_ratio_20'] = p.turnover/roll('turnover', 'mean')
    p['dist_high20'] = c/roll('high', 'max', m=1) - 1
    p['dist_low20'] = c/roll('low', 'min', m=1) - 1
    p['range_fraction'] = (p.high - p.low)/c
    p['intraday_return'] = c/p.open - 1
    p['open_return'] = p.open/prev - 1
    p['vwap_deviation'] = p.turnover/p.volume/c - 1
    p['gap_days'] = g['date'].diff().dt.days
    p['log_turnover'] = np.log1p(p.turnover)
    p['log_trade_count'] = np.log1p(p.trade_count)
    p['thin'] = (p.trade_count < 20).astype(float)

    mkt = (p.groupby('date').agg(mkt_ret=('ret_1', 'mean'), mkt_turnover=('turnover', 'sum'))
             .join(p.assign(up=p.ret_1 > 0, seen=p.ret_1.notna()).groupby('date')[['up', 'seen']].sum()))
    mkt['breadth'] = mkt.up/mkt.seen
    index = (1 + mkt.mkt_ret.fillna(0)).cumprod()
    mkt['mkt_ret_5'] = index/index.shift(5) - 1
    lt = np.log(mkt.mkt_turnover)
    mkt['mkt_turnover_z'] = (lt - lt.rolling(20, min_periods=10).mean())/lt.rolling(20, min_periods=10).std()
    p = p.join(mkt[['mkt_ret', 'mkt_ret_5', 'breadth', 'mkt_turnover_z']], on='date')
    p['excess_ret_1'] = p.ret_1 - p.mkt_ret
    p['excess_ret_5'] = p.ret_5 - p.mkt_ret_5
    p['excess_sector_ret'] = p.ret_1 - p.groupby(['date', 'sector'], observed=True).ret_1.transform('mean')

    p['broker_imbalance'] = (p.top5_net_buy + p.top5_net_sell)/p.volume
    p['hhi_diff'] = p.buyer_hhi - p.seller_hhi
    g = p.groupby('symbol', sort=False)
    p['imbalance_5'] = roll('broker_imbalance', 'mean', 5, 1)
    p['hhi_diff_5'] = roll('hhi_diff', 'mean', 5, 1)
    p['crossed_share'] = p.crossed_volume/p.volume
    p['log_buyer_seller_ratio'] = np.log(p.n_buyers/p.n_sellers)
    p['log_n_buyers'] = np.log(p.n_buyers)

    p['fwd_return'] = g['close'].shift(-horizon)/c - 1
    p['label_end'] = g['date'].shift(-horizon)
    p['threshold'] = np.maximum(min_move, k_vol*p.vol_20*np.sqrt(horizon))
    p['target'] = np.select([p.fwd_return < -p.threshold, p.fwd_return > p.threshold], [0, 2], 1)
    p.loc[p.fwd_return.isna(), 'target'] = -1
    p['pos'] = g.cumcount()
    numeric = TECH + MARKET + BROKER + ['promoter']
    p[numeric] = p[numeric].replace([np.inf, -np.inf], np.nan)
    return p.drop(columns=['_up', '_down', '_macd'])
