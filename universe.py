from __future__ import annotations
import ccxt
import pandas as pd
import numpy as np

def _fetch_tickers(ex, quote: str):
    tickers = ex.fetch_tickers()
    rows = []
    for sym, t in tickers.items():
        try:
            if not sym.endswith('/' + quote):
                continue
            last = t.get('last') or t.get('close')
            base_vol = t.get('baseVolume')
            quote_vol = t.get('quoteVolume')
            if quote_vol is None and base_vol is not None and last is not None:
                quote_vol = base_vol * last
            rows.append({'symbol': sym, 'last': last, 'quoteVolume': quote_vol})
        except Exception:
            continue
    return pd.DataFrame(rows)

def _daily_volatility(ex, symbol: str, days: int) -> float:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe='1d', limit=days+1)
    if not ohlcv or len(ohlcv) < 2:
        return 0.0
    df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
    ret = pd.Series(df['c']).pct_change().dropna()
    return float(ret.std() * 100.0)

def _trend_ok(ex, symbol: str, tf: str, ma: int, allow_downtrend: bool):
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=max(ma+2, 60))
    df = pd.DataFrame(ohlcv, columns=['ts','o','h','l','c','v'])
    ma_series = pd.Series(df['c']).rolling(ma).mean()
    price = float(df['c'].iloc[-1])
    ma_val = float(ma_series.iloc[-1])
    if pd.isna(ma_val):
        return False
    if allow_downtrend:
        return True
    return price > ma_val

def build_universe(cfg) -> list[str]:
    mode = cfg.get('universe', {}).get('mode', 'static')
    if mode == 'static':
        return cfg.get('universe', {}).get('symbols', [])
    d = cfg.get('universe', {}).get('dynamic', {})
    quote = d.get('quote', 'USDT')
    top_n = int(d.get('top_n', 12))
    exclude = set(d.get('exclude', []) or [])
    min_vol_usd = float(d.get('min_vol_usd', 0))
    min_price = float(d.get('min_price', 0))
    vol_days = int(d.get('volatility_days', 14))
    min_volatility_pct = float(d.get('min_volatility_pct', 0))
    trend_tf = d.get('trend_tf', '4h')
    trend_ma = int(d.get('trend_ma', 50))
    allow_downtrend = bool(d.get('allow_downtrend', False))

    ex_id = cfg.get('exchange', 'binance')
    ex = getattr(ccxt, ex_id)()

    df_t = _fetch_tickers(ex, quote)
    if df_t.empty:
        return []

    df_t = df_t.dropna(subset=['last']).copy()
    df_t = df_t[df_t['last'] >= min_price]
    df_t = df_t[~df_t['symbol'].isin(exclude)]
    df_t['quoteVolume'] = df_t['quoteVolume'].fillna(0.0)
    df_t = df_t[df_t['quoteVolume'] >= min_vol_usd]

    vol_vals = []
    trend_flags = []
    for sym in df_t['symbol'].tolist():
        try:
            v = _daily_volatility(ex, sym, vol_days)
            vol_vals.append(v)
            t_ok = _trend_ok(ex, sym, trend_tf, trend_ma, allow_downtrend)
            trend_flags.append(t_ok)
        except Exception:
            vol_vals.append(0.0)
            trend_flags.append(False)
    df_t['volatility_pct'] = vol_vals
    df_t['trend_ok'] = trend_flags

    if min_volatility_pct > 0:
        df_t = df_t[df_t['volatility_pct'] >= min_volatility_pct]
    if not df_t.empty:
        df_t = df_t[df_t['trend_ok']]

    df_t = df_t.sort_values('quoteVolume', ascending=False)
    return df_t['symbol'].head(top_n).tolist()
