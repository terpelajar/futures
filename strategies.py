from __future__ import annotations
import pandas as pd
import numpy as np

# =========================
# Helper indicators
# =========================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(length).mean()
    roll_down = pd.Series(down, index=close.index).rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def _std(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).std(ddof=0)

def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52):
    high, low = df['high'], df['low']
    conversion = (high.rolling(conv).max() + low.rolling(conv).min()) / 2
    base_line = (high.rolling(base).max() + low.rolling(base).min()) / 2
    leading_span_a = ((conversion + base_line) / 2).shift(base)
    leading_span_b = ((high.rolling(span_b).max() + low.rolling(span_b).min()) / 2).shift(base)
    lagging_span = df['close'].shift(-base)
    return conversion, base_line, leading_span_a, leading_span_b, lagging_span

def _cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    mfm = ((close - low) - (high - close)) / ((high - low).replace(0, np.nan))
    mfv = mfm * volume
    cmf = mfv.rolling(length).sum() / volume.rolling(length).sum()
    return cmf.fillna(0)

def _vwap(df: pd.DataFrame, length: int = None) -> pd.Series:
    typical = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, np.nan)
    if length is None:
        cum_pv = (typical * vol).cumsum()
        cum_vol = vol.cumsum().replace(0, np.nan)
        return cum_pv / cum_vol
    else:
        pv = (typical * vol).rolling(length).sum()
        v = vol.rolling(length).sum().replace(0, np.nan)
        return pv / v

# =========================
# Strategy signal generator
# =========================
def generate_signals(df: pd.DataFrame, name: str, params: dict) -> pd.DataFrame:
    df = df.copy()
    if not {'open','high','low','close','volume'}.issubset(df.columns):
        raise ValueError("DataFrame must have OHLCV columns.")

    # Precompute shared indicators
    if name in ('macd_cross','ichimoku_macd','combo_rsi_bb_macd'):
        f = int(params.get('macd_fast', 12))
        s = int(params.get('macd_slow', 26))
        sig = int(params.get('macd_signal', 9))
        macd, signal, hist = _macd(df['close'], f, s, sig)
        df['macd'], df['macd_sig'] = macd, signal

    if name in ('ichimoku_macd','ichimoku_cmf'):
        conv = int(params.get('ichimoku_conversion', 9))
        base = int(params.get('ichimoku_base', 26))
        span_b = int(params.get('ichimoku_span_b', 52))
        conv_s, base_s, span_a, span_b_s, lag = _ichimoku(df, conv, base, span_b)
        df['tenkan'], df['kijun'], df['span_a'], df['span_b'] = conv_s, base_s, span_a, span_b_s

    if name in ('ichimoku_cmf',):
        df['cmf'] = _cmf(df, int(params.get('cmf_len', 20)))

    if name in ('combo_rsi_bb_macd', 'rsi_reversal', 'boll_breakout'):
        df['rsi'] = _rsi(df['close'], int(params.get('rsi_len', 14)))

    # --- SMA cross
    if name == 'sma_cross':
        s = int(params.get('sma_short', 10)); l = int(params.get('sma_long', 30))
        df['sma_s'] = _sma(df['close'], s); df['sma_l'] = _sma(df['close'], l)
        df['signal'] = (df['sma_s'] > df['sma_l']).astype(int).replace(0, -1).shift(1).fillna(0)

    # --- EMA cross
    elif name == 'ema_cross':
        s = int(params.get('ema_short', 12)); l = int(params.get('ema_long', 26))
        df['ema_s'] = _ema(df['close'], s); df['ema_l'] = _ema(df['close'], l)
        df['signal'] = (df['ema_s'] > df['ema_l']).astype(int).replace(0, -1).shift(1).fillna(0)

    # --- RSI reversal
    elif name == 'rsi_reversal':
        buy = float(params.get('rsi_buy', 30)); sell = float(params.get('rsi_sell', 70))
        sig = np.where(df['rsi'] < buy, 1, np.where(df['rsi'] > sell, -1, 0))
        df['signal'] = pd.Series(sig, index=df.index).shift(1).fillna(0)

    # --- Bollinger breakout (dengan filter tren & CMF opsional, short opsional)
    elif name == 'boll_breakout':
        n = int(params.get('bb_len', 20)); k = float(params.get('bb_std', 2.0))
        ma = _sma(df['close'], n); std = _std(df['close'], n)
        df['bb_upper'], df['bb_lower'], df['bb_mid'] = ma + k*std, ma - k*std, ma

        # breakout crosses
        cond_up = (df['close'].shift(1) <= df['bb_upper'].shift(1)) & (df['close'] > df['bb_upper'])
        cond_dn = (df['close'].shift(1) >= df['bb_lower'].shift(1)) & (df['close'] < df['bb_lower'])

        # Filters
        trend_ma = int(params.get('bb_trend_ma', 0))  # 0 = nonaktif
        require_cmf = bool(params.get('bb_require_cmf', False))
        allow_shorts = bool(params.get('bb_allow_shorts', True))
        cmf_len = int(params.get('cmf_len', 20))

        if trend_ma > 0:
            trend_ma_series = _sma(df['close'], trend_ma)
            trend_long = df['close'] > trend_ma_series
            trend_short = df['close'] < trend_ma_series
        else:
            trend_long = pd.Series(True, index=df.index)
            trend_short = pd.Series(True, index=df.index)

        if require_cmf:
            df['cmf'] = _cmf(df, cmf_len)
            cmf_long = (df['cmf'] > 0)
            cmf_short = (df['cmf'] < 0)
        else:
            cmf_long = pd.Series(True, index=df.index)
            cmf_short = pd.Series(True, index=df.index)

        long_cond = cond_up & trend_long & cmf_long
        short_cond = cond_dn & trend_short & cmf_short if allow_shorts else pd.Series(False, index=df.index)

        sig = np.where(long_cond, 1, np.where(short_cond, -1, 0))
        df['signal'] = pd.Series(sig, index=df.index).shift(1).fillna(0)

    # --- MACD cross (selalu bisa 1/-1)
    elif name == 'macd_cross':
        sig = np.where(df['macd'] > df['macd_sig'], 1, -1)
        df['signal'] = pd.Series(sig, index=df.index).shift(1).fillna(0)

    # --- High/Low breakout (kedua arah)
    elif name == 'breakout_hhll':
        lb = int(params.get('breakout_lookback', 20))
        hh = df['high'].rolling(lb).max(); ll = df['low'].rolling(lb).min()
        buy = (df['close'] > hh.shift(1)); sell = (df['close'] < ll.shift(1))
        sig = np.where(buy, 1, np.where(sell, -1, np.nan))
        df['signal'] = pd.Series(sig, index=df.index).ffill().fillna(0)

    # --- VWAP revert (kedua arah)
    elif name == 'vwap_revert':
        vw = _vwap(df)
        sig = np.where(df['close'] < vw, 1, np.where(df['close'] > vw, -1, 0))
        df['vwap'] = vw
        df['signal'] = pd.Series(sig, index=df.index).shift(1).fillna(0)

    # --- Ichimoku + MACD (sekarang simetris long/short, dengan exit)
    elif name == 'ichimoku_macd':
        allow_shorts = bool(params.get('ichimoku_allow_shorts', True))
        close = df['close']
        span_max = df[['span_a','span_b']].max(axis=1)
        span_min = df[['span_a','span_b']].min(axis=1)

        above_cloud = (close > span_max)
        below_cloud = (close < span_min)

        tenkan_cross_up   = (df['tenkan'].shift(1) <= df['kijun'].shift(1)) & (df['tenkan'] > df['kijun'])
        tenkan_cross_down = (df['tenkan'].shift(1) >= df['kijun'].shift(1)) & (df['tenkan'] < df['kijun'])

        macd_up = df['macd'] > df['macd_sig']
        macd_down = df['macd'] < df['macd_sig']

        long_entry  = above_cloud & tenkan_cross_up   & macd_up
        long_exit   = (macd_down | (close < df['kijun']))

        short_entry = allow_shorts & below_cloud & tenkan_cross_down & macd_down
        short_exit  = (macd_up | (close > df['kijun']))

        s = pd.Series(np.nan, index=df.index, dtype=float)
        s[long_entry] = 1
        s[short_entry] = -1
        s[long_exit | short_exit] = 0
        df['signal'] = s.ffill().fillna(0).astype(int)

    # --- Ichimoku + CMF (sekarang simetris long/short, dengan exit)
    elif name == 'ichimoku_cmf':
        allow_shorts = bool(params.get('ichimoku_allow_shorts', True))
        close = df['close']
        span_max = df[['span_a','span_b']].max(axis=1)
        span_min = df[['span_a','span_b']].min(axis=1)

        above_cloud = (close > span_max)
        below_cloud = (close < span_min)

        tenkan_cross_up   = (df['tenkan'].shift(1) <= df['kijun'].shift(1)) & (df['tenkan'] > df['kijun'])
        tenkan_cross_down = (df['tenkan'].shift(1) >= df['kijun'].shift(1)) & (df['tenkan'] < df['kijun'])

        cmf_pos = (df['cmf'] > 0)
        cmf_neg = (df['cmf'] < 0)

        long_entry  = above_cloud & tenkan_cross_up   & cmf_pos
        long_exit   = (cmf_neg | (close < df['kijun']))

        short_entry = allow_shorts & below_cloud & tenkan_cross_down & cmf_neg
        short_exit  = (cmf_pos | (close > df['kijun']))

        s = pd.Series(np.nan, index=df.index, dtype=float)
        s[long_entry] = 1
        s[short_entry] = -1
        s[long_exit | short_exit] = 0
        df['signal'] = s.ffill().fillna(0).astype(int)

    else:
        raise ValueError(f"Unknown strategy name: {name}")

    return df
