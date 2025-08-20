# live_trader_umf.py
from __future__ import annotations
import time, math, yaml, traceback
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
from binance.um_futures import UMFutures
from telegram import Bot

from strategies import generate_signals, _atr  # pakai file kamu

# ======================
# Utils & helpers
# ======================
def now_utc_str():
    return datetime.now(timezone.utc).strftime('%H:%M:%S')

def log(level: str, msg: str):
    print(f"[{now_utc_str()}] {level:<6} | {msg}", flush=True)

def sfloat(x, d=0.0):
    try: return float(x)
    except Exception: return float(d)

def sint(x, d=0):
    try: return int(x)
    except Exception: return int(d)

def norm_sym(sym: str) -> str:
    # "BTC/USDT" -> "BTCUSDT"
    return sym.replace("/", "").upper().strip()

def denorm_sym(sym: str) -> str:
    # "BTCUSDT" -> "BTC/USDT"
    return f"{sym[:-4]}/{sym[-4:]}" if sym.endswith("USDT") else sym

def last_valid_float(series: pd.Series, lookback: int = 5) -> float | None:
    s = pd.to_numeric(series.tail(lookback), errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else None

# ======================
# Market data
# ======================
def fetch_klines_df(client: UMFutures, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    # symbol "BTCUSDT", interval "15m"
    ks = client.klines(symbol=symbol, interval=interval, limit=int(limit), recvWindow=5000)
    # kline fields: [openTime, o, h, l, c, v, closeTime, q, numTrades, takerBase, takerQuote, ignore]
    cols = ['open_time','open','high','low','close','volume','close_time','q','n','tb','tq','i']
    df = pd.DataFrame(ks, columns=cols)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df['timestamp'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
    df = df.set_index('timestamp')[['open','high','low','close','volume']].dropna()
    return df

def fetch_mark_price(client: UMFutures, symbol: str) -> float | None:
    try:
        mp = client.mark_price(symbol=symbol)  # {'symbol':'BTCUSDT','markPrice':'...'}
        return sfloat(mp.get('markPrice'), None)
    except Exception:
        return None

# ======================
# Exchange info / filters
# ======================
def load_exchange_info(client: UMFutures) -> dict:
    ei = client.exchange_info()  # futures exchangeInfo
    symbols = ei.get('symbols', [])
    by_symbol = {s['symbol']: s for s in symbols}
    return by_symbol

def filter_usdtm_symbols(watch: list[str], exinfo: dict) -> tuple[list[str], list[tuple[str,str]]]:
    ok, dropped = [], []
    for raw in watch:
        s = norm_sym(raw)
        m = exinfo.get(s)
        if not m:
            dropped.append((raw, "not listed")); continue
        # futures perpetual, quote=USDT, trading
        if m.get('contractType') != 'PERPETUAL':
            dropped.append((raw, "not PERPETUAL")); continue
        if m.get('quoteAsset') != 'USDT':
            dropped.append((raw, "quote!=USDT")); continue
        if m.get('status') != 'TRADING':
            dropped.append((raw, "status!=TRADING")); continue
        ok.append(s)  # pakai format tanpa slash
    return ok, dropped

def get_step_minqty_from_filters(exinfo: dict, symbol: str) -> tuple[float|None, float|None]:
    """Ambil LOT_SIZE.stepSize dan minQty untuk pembulatan kuantitas."""
    f = exinfo.get(symbol, {}).get('filters', [])
    step, minqty = None, None
    for flt in f:
        if flt.get('filterType') in ('LOT_SIZE', 'MARKET_LOT_SIZE'):
            step = sfloat(flt.get('stepSize'), None)
            mq = sfloat(flt.get('minQty'), None)
            if step is not None: 
                # pilih yang paling ketat
                minqty = mq if (minqty is None or (mq is not None and mq > minqty)) else minqty
    return step, minqty

def quantize_qty(symbol: str, qty: float, price: float, exinfo: dict) -> float:
    step, minqty = get_step_minqty_from_filters(exinfo, symbol)
    if qty <= 0:
        return 0.0
    q = float(qty)
    if step and step > 0:
        q = math.floor(q / step) * step
    if minqty and q < minqty:
        q = 0.0
    return float(q)

# ======================
# Risk / sizing / levels
# ======================
def qty_from_risk(entry: float, sl: float, equity_usdt: float, rpt: float) -> tuple[float, float]:
    stop_abs = abs(entry - sl)
    risk_amount = equity_usdt * rpt
    if stop_abs <= 0 or risk_amount <= 0: 
        return 0.0, 0.0
    return risk_amount / stop_abs, risk_amount

def compute_levels(df: pd.DataFrame, side: int, params: dict, risk: dict, entry_source: str = "close"):
    price = df['open'] if entry_source == "open" else df['close']
    entry = last_valid_float(price, 5)
    if entry is None:
        raise ValueError("no valid entry")
    rr = sfloat(risk.get('take_profit_rr', 2.0), 2.0)
    stop_mode = (risk.get('stop_mode') or 'atr').lower()
    if stop_mode == 'atr':
        ap = sint(params.get('atr_entry_period', 14), 14)
        am = sfloat(params.get('atr_entry_mult', 1.5), 1.5)
        atr = last_valid_float(_atr(df, ap), 5)
        if not atr or atr <= 0:
            raise ValueError("ATR unavailable")
        if side > 0:
            sl = entry - am * atr
            tp = entry + rr * (entry - sl)
        else:
            sl = entry + am * atr
            tp = entry - rr * (sl - entry)
    else:
        slp = sfloat(risk.get('stop_loss_pct', 0.02), 0.02)
        tpp = risk.get('take_profit_pct')
        tpp = sfloat(tpp if tpp is not None else rr * slp, rr * slp)
        sl = entry * (1 - slp) if side > 0 else entry * (1 + slp)
        tp = entry * (1 + tpp) if side > 0 else entry * (1 - tpp)
    return float(entry), float(sl), float(tp)

# ----- live margin/bracket clamps -----
def futures_free_usdt(client: UMFutures) -> float:
    """availableBalance USDT di wallet futures."""
    try:
        bal = client.balance(recvWindow=5000)  # list of assets
        for a in bal:
            if a.get('asset') == 'USDT':
                return sfloat(a.get('availableBalance'), 0.0)
    except Exception as e:
        log("WARN", f"balance: {e}")
    return 0.0

def get_notional_cap(client: UMFutures, symbol: str, leverage: int) -> float:
    """
    Ambil notionalCap terbesar yang masih valid untuk leverage yang diajukan.
    """
    try:
        data = client.leverage_bracket(symbol=symbol, recvWindow=5000)
        # response bisa list, ambil item 0
        item = data[0] if isinstance(data, list) and data else data
        brackets = item.get('brackets', [])
        caps_ok, caps_all = [], []
        for b in brackets:
            ilv = sint(b.get('initialLeverage'), 0)
            cap = sfloat(b.get('notionalCap'), 0.0)
            caps_all.append(cap)
            if ilv >= leverage:
                caps_ok.append(cap)
        if caps_ok:
            return max(caps_ok)
        if caps_all:
            return min(caps_all)
        return float('inf')
    except Exception as e:
        log("WARN", f"leverage_bracket {symbol}: {e}")
        return float('inf')

def clamp_qty_by_free_and_bracket(client: UMFutures, exinfo: dict, symbol: str,
                                  qty_init: float, entry: float, lev: int,
                                  margin_safety=0.95, bracket_safety=0.98) -> tuple[float, dict]:
    info = {}
    notional_init = float(qty_init) * float(entry)
    free = futures_free_usdt(client)
    cap  = get_notional_cap(client, symbol, lev)
    allowed_free_notional = free * lev * margin_safety
    allowed_bracket_notional = (cap * bracket_safety) if cap != float('inf') else float('inf')
    allowed_notional = min(notional_init, allowed_free_notional, allowed_bracket_notional)
    info.update(dict(
        free_usdt=free,
        cap_notional=cap,
        notional_init=notional_init,
        allowed_free_notional=allowed_free_notional,
        allowed_bracket_notional=allowed_bracket_notional,
        allowed_notional=allowed_notional,
    ))
    if allowed_notional <= 0:
        return 0.0, info
    qty_target = allowed_notional / max(entry, 1e-12)
    qty_final = quantize_qty(symbol, qty_target, entry, exinfo)
    return float(qty_final), info

# ======================
# Account/futures settings
# ======================
def ensure_settings(client: UMFutures, symbol: str, leverage: int, margin_type: str, position_mode: str):
    # position side (dual or oneway)
    try:
        dual = 'true' if position_mode.lower() == 'hedge' else 'false'
        client.change_position_mode(dualSidePosition=dual, recvWindow=5000)
    except Exception:
        pass
    # margin type
    try:
        client.change_margin_type(symbol=symbol, marginType=margin_type.upper(), recvWindow=5000)
    except Exception:
        pass
    # leverage
    try:
        client.change_leverage(symbol=symbol, leverage=int(leverage), recvWindow=5000)
    except Exception:
        pass

# ======================
# Orders
# ======================
def cancel_all_open_orders(client: UMFutures, symbol: str):
    try:
        client.cancel_open_orders(symbol=symbol, recvWindow=5000)
    except Exception as e:
        log("WARN", f"cancel_open_orders {denorm_sym(symbol)}: {e}")

def place_market_order(client: UMFutures, symbol: str, side: int, qty: float, position_mode: str):
    side_txt = 'BUY' if side > 0 else 'SELL'
    params = dict(recvWindow=5000)
    if position_mode.lower() == 'hedge':
        params['positionSide'] = 'LONG' if side > 0 else 'SHORT'
    return client.new_order(symbol=symbol, side=side_txt, type='MARKET', quantity=qty, **params)

def place_sl_tp(client: UMFutures, symbol: str, side: int, sl: float, tp: float, working_type: str, position_mode: str):
    base = dict(recvWindow=5000, workingType=working_type, reduceOnly='true', closePosition='true', timeInForce='GTC')
    if position_mode.lower() == 'hedge':
        base_long  = {**base, 'positionSide': 'LONG'}
        base_short = {**base, 'positionSide': 'SHORT'}
        sl_params = base_long if side > 0 else base_short
        tp_params = base_long if side > 0 else base_short
    else:
        sl_params = dict(base)
        tp_params = dict(base)

    sl_side = 'SELL' if side > 0 else 'BUY'
    tp_side = 'SELL' if side > 0 else 'BUY'

    # SL
    try:
        client.new_order(symbol=symbol, side=sl_side, type='STOP_MARKET', stopPrice=sl, quantity=None, **sl_params)
    except Exception as e:
        log("ERROR", f"place SL {denorm_sym(symbol)}: {e}")
    # TP
    try:
        client.new_order(symbol=symbol, side=tp_side, type='TAKE_PROFIT_MARKET', stopPrice=tp, quantity=None, **tp_params)
    except Exception as e:
        log("ERROR", f"place TP {denorm_sym(symbol)}: {e}")

def get_open_orders(client: UMFutures, symbol: str):
    try:
        return client.get_open_orders(symbol=symbol, recvWindow=5000)
    except Exception:
        return []

def cancel_only_sl(client: UMFutures, symbol: str):
    """Batalkan hanya STOP/STOP_MARKET closePosition reduceOnly."""
    try:
        orders = get_open_orders(client, symbol)
        for od in orders:
            t = od.get('type', '')
            cp = str(od.get('closePosition', '')).lower() in ('true','True','1')
            ro = str(od.get('reduceOnly', '')).lower() in ('true','True','1')
            if t in ('STOP', 'STOP_MARKET') and (cp or ro):
                client.cancel_order(symbol=symbol, orderId=od.get('orderId'), recvWindow=5000)
    except Exception as e:
        log("WARN", f"cancel_only_sl {denorm_sym(symbol)}: {e}")

# ======================
# Breakeven @ +1R
# ======================
def maybe_move_to_breakeven(client: UMFutures, symbol: str, side: int, entry: float, sl: float,
                            be_cfg: dict, working_type: str, position_mode: str):
    """
    Jika harga sudah ≥ entry + 1R (long) atau ≤ entry - 1R (short),
    geser SL ke BE + fee/slip buffer.
    """
    if not be_cfg or not be_cfg.get('enabled', True):
        return

    r_multiple = sfloat(be_cfg.get('r_multiple', 1.0), 1.0)
    fee_taker_pct = sfloat(be_cfg.get('fee_taker_pct', 0.0005), 0.0005)
    slip_pct = sfloat(be_cfg.get('slippage_est_pct', 0.0002), 0.0002)
    extra = sfloat(be_cfg.get('extra_buffer_pct', 0.0001), 0.0001)

    # total buffer "per sisi" (open + close + slip2x + extra)
    total = (fee_taker_pct * 2.0) + (slip_pct * 2.0) + extra

    # 1R:
    R = abs(entry - sl)

    # Harga acuan -> mark price
    mp = fetch_mark_price(client, symbol)
    if mp is None:
        return

    hit = False
    if side > 0:
        # long: profit >= r_multiple * R ?
        if (mp - entry) >= (r_multiple * R):
            new_sl = entry * (1.0 + total)
            hit = True
    else:
        if (entry - mp) >= (r_multiple * R):
            new_sl = entry * (1.0 - total)
            hit = True

    if not hit:
        return

    # Geser SL: batalkan SL lama saja, lalu pasang STOP_MARKET baru
    try:
        cancel_only_sl(client, symbol)
        # pasang SL baru @BE+fee
        base = dict(recvWindow=5000, workingType=working_type, reduceOnly='true', closePosition='true', timeInForce='GTC')
        if position_mode.lower() == 'hedge':
            base['positionSide'] = 'LONG' if side > 0 else 'SHORT'
        sl_side = 'SELL' if side > 0 else 'BUY'
        client.new_order(symbol=symbol, side=sl_side, type='STOP_MARKET', stopPrice=new_sl, quantity=None, **base)
        log("MOVE", f"{denorm_sym(symbol)} -> BE SL @ {new_sl:.6f}")
    except Exception as e:
        log("WARN", f"move_to_BE {denorm_sym(symbol)}: {e}")

# ======================
# Snapshot helper
# ======================
def send_snapshot(bot: Bot, chat_ids, client: UMFutures, symbols: list[str], tf: str, lookback: int,
                  default_strategy: str, strat_map: dict, base_params: dict, params_over: dict, risk: dict):
    lines = []
    for s in symbols:
        try:
            df = fetch_klines_df(client, s, tf, lookback)
            if len(df) < 60:
                lines.append(f"{denorm_sym(s)}: data<60 bars")
                continue
            strat = strat_map.get(denorm_sym(s), default_strategy)
            params = dict(base_params); params.update(params_over.get(denorm_sym(s), {}))
            sig_df = generate_signals(df, strat, params)
            sig = int(sig_df['signal'].iloc[-1])
            entry, sl, tp = compute_levels(df, (sig if sig != 0 else 1), params, risk, entry_source=str(risk.get('entry_source','close')))
            if sig > 0: tp_pct = (tp/entry - 1.0) * 100.0
            elif sig < 0: tp_pct = (1.0 - tp/entry) * 100.0
            else: tp_pct = 0.0
            lines.append(f"{denorm_sym(s)}: sig={sig}  TP≈{abs(tp_pct):.2f}%")
        except Exception as e:
            lines.append(f"{denorm_sym(s)}: err {str(e)[:60]}")
    msg = "📸 Snapshot (latest signals)\n" + "\n".join(lines[:20])
    for cid in chat_ids:
        try: bot.send_message(chat_id=cid, text=msg)
        except Exception: pass

# ======================
# Main
# ======================
def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exch = (cfg.get('exchange') or 'binance').lower()
    if exch != 'binance':
        raise RuntimeError("Script ini khusus Binance USDⓈ-M Futures")

    market_type = (cfg.get('market_type') or 'futures').lower()
    if market_type not in ('futures','future'):
        raise RuntimeError("Gunakan market_type: futures (USDⓈ-M)")

    tf = cfg['timeframe']
    lookback = sint(cfg.get('lookback_candles', 300), 300)

    # universe (boleh "ETH/USDT" dsb)
    syms_cfg = (cfg.get('universe', {}) or {}).get('symbols', []) or cfg.get('symbols', [])
    if not syms_cfg:
        raise RuntimeError("No symbols in config.yaml")
    symbols_raw = syms_cfg[:]  # keep original for map keys
    symbols = [norm_sym(s) for s in syms_cfg]

    # telegram
    tg = cfg['telegram']
    bot = Bot(token=tg['bot_token'])
    chat_ids = tg['chat_ids']

    # risk & live cfg
    risk = cfg.get('risk', {})
    leverage = sint(risk.get('leverage', 3), 3)
    rpt = sfloat(risk.get('risk_per_trade', 0.01), 0.01)
    max_margin_frac = sfloat(risk.get('max_margin_frac', 0.2), 0.2)  # batas porsi margin pseudo
    entry_source = str(risk.get('entry_source', 'close'))
    be_cfg = (risk.get('breakeven') or {})

    live = (cfg.get('live') or {})
    testnet = bool(live.get('testnet', True))
    position_mode = live.get('position_mode', 'oneway')
    margin_type   = live.get('margin_type', 'ISOLATED')
    working_type  = live.get('working_type', 'MARK_PRICE')
    send_fills    = bool(live.get('send_fills_to_telegram', True))
    poll_sleep    = sint(live.get('poll_sleep', 10), 10)
    hb_minutes    = sint(live.get('heartbeat_minutes', 5), 5)

    # strategies
    default_strategy = (cfg.get('strategies') or ['boll_breakout'])[0]
    strat_map   = cfg.get('strategy_per_symbol', {}) or {}
    base_params = cfg.get('params', {}) or {}
    params_over = cfg.get('params_overrides', {}) or {}

    # equity pseudo (cadangan kalau mau batasi porsi margin di atas)
    equity_usdt = sfloat(cfg.get('account', {}).get('equity_usdt', 1000.0), 1000.0)

    # --- client
    keys = ((cfg.get('keys') or {}).get('binance') or {})
    api_key = keys.get('api_key')
    api_sec = keys.get('secret')
    if not api_key or not api_sec:
        raise RuntimeError("API key/secret kosong di config.yaml -> keys.binance")

    base_url = "https://testnet.binancefuture.com" if testnet else None
    client = UMFutures(key=api_key, secret=api_sec, base_url=base_url) if base_url else UMFutures(key=api_key, secret=api_sec)

    # --- exchangeInfo & filter
    exinfo = load_exchange_info(client)
    ok, dropped = filter_usdtm_symbols(symbols, exinfo)
    if dropped:
        log("FILTER", "dropped " + ", ".join([d[0] for d in dropped][:20]) + ("..." if len(dropped) > 20 else ""))
    if not ok:
        fallback = [s for s in ("BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","ADAUSDT") if s in exinfo]
        if not fallback:
            raise RuntimeError("No valid symbols after filter")
        ok = fallback
        log("FILTER", "using fallback " + ", ".join([denorm_sym(s) for s in ok]))
    symbols = ok

    # --- ensure futures settings per symbol
    for s in symbols:
        ensure_settings(client, s, leverage, margin_type, position_mode)

    # --- startup ping
    start_text = (
        "🚀 LIVE Trader started\n"
        f"Exchange: binance (futures {'TESTNET' if testnet else 'MAINNET'})\n"
        f"TF: {tf}  Symbols: {len(symbols)}  Leverage: {leverage}x  Mode: {position_mode}/{margin_type}\n"
        f"workingType={working_type}, risk_per_trade={rpt*100:.2f}%"
    )
    if send_fills:
        for cid in chat_ids:
            try: bot.send_message(chat_id=cid, text=start_text)
            except Exception: pass
    log("START", start_text.replace("\n"," | "))

    # snapshot sekali
    send_snapshot(bot, chat_ids, client, symbols, tf, lookback, default_strategy, strat_map, base_params, params_over, risk)

    # state
    last_ts = {s: None for s in symbols}
    last_sig = {s: 0 for s in symbols}
    pos_state = {s: None for s in symbols}  # { 'side':1/-1, 'entry':.., 'sl':.., 'tp':.. }
    next_hb = datetime.now(timezone.utc) + timedelta(minutes=hb_minutes)

    # --- loop
    while True:
        processed = 0
        try:
            for s in symbols:
                try:
                    df = fetch_klines_df(client, s, tf, lookback)
                    processed += 1
                    if len(df) < 60:
                        continue
                    newest = df.index[-1]
                    if last_ts[s] is None or newest > last_ts[s]:
                        last_ts[s] = newest

                        # generate signal
                        key = denorm_sym(s)
                        strat = strat_map.get(key, default_strategy)
                        params = dict(base_params); params.update(params_over.get(key, {}))
                        sig_df = generate_signals(df, strat, params)
                        sig = int(sig_df['signal'].iloc[-1])

                        # BE manager (kalau ada posisi)
                        st = pos_state.get(s)
                        if st:
                            maybe_move_to_breakeven(client, s, st['side'], st['entry'], st['sl'], be_cfg, working_type, position_mode)

                        # Entry hanya ketika berubah arah / dari flat->posisi
                        if sig != 0 and last_sig.get(s, 0) != sig:
                            # hitung level & qty risk-based
                            entry, sl, tp = compute_levels(df, sig, params, risk, entry_source=entry_source)

                            # sizing risk pseudo-equity (optional batas awal)
                            qty_risk, _ = qty_from_risk(entry, sl, equity_usdt, rpt)
                            qty_init = quantize_qty(s, qty_risk, entry, exinfo)
                            if qty_init <= 0:
                                last_sig[s] = sig
                                log("SKIP", f"{denorm_sym(s)} qty<=0 (risk)")
                                continue

                            # clamp by real free & bracket
                            qty, clamp = clamp_qty_by_free_and_bracket(
                                client, exinfo, s, qty_init, entry, leverage,
                                margin_safety=0.95, bracket_safety=0.98
                            )
                            if qty <= 0:
                                last_sig[s] = sig
                                log("SKIP", f"{denorm_sym(s)} qty=0 after clamp | free={clamp.get('free_usdt'):.2f} cap={clamp.get('cap_notional')}")
                                continue

                            notional = qty * entry
                            # tambahan: batasi porsi margin semu (opsional)
                            max_margin = equity_usdt * max_margin_frac
                            margin_used = notional / max(leverage,1)
                            if margin_used > max_margin and notional > 0:
                                scale = max_margin / margin_used
                                qty = quantize_qty(s, qty * scale, entry, exinfo)
                                notional = qty * entry
                                margin_used = notional / max(leverage,1)
                            if qty <= 0:
                                last_sig[s] = sig
                                log("SKIP", f"{denorm_sym(s)} qty after max_margin clamp <= 0")
                                continue

                            # cancel semua open-orders lama
                            cancel_all_open_orders(client, s)

                            # market order
                            try:
                                place_market_order(client, s, sig, qty, position_mode)
                            except Exception as e:
                                last_sig[s] = sig
                                err = f"⚠️ Order gagal\n{denorm_sym(s)} {'LONG' if sig>0 else 'SHORT'}\nQty≈ {qty:.6f}\nErr: {str(e)[:180]}"
                                log("ERROR", f"order {denorm_sym(s)}: {e}")
                                if send_fills:
                                    for cid in chat_ids:
                                        try: bot.send_message(chat_id=cid, text=err)
                                        except Exception: pass
                                # backoff simple utk rate-limit
                                es = str(e)
                                if '429' in es or '418' in es or 'Too many requests' in es:
                                    log("RATE", "Backoff 60s karena rate limit"); time.sleep(60)
                                continue

                            # protective orders
                            place_sl_tp(client, s, sig, sl, tp, working_type, position_mode)

                            # simpan state posisi minimal (buat BE)
                            pos_state[s] = {'side': sig, 'entry': entry, 'sl': sl, 'tp': tp}

                            # notif
                            side_tag = "🟢 LONG" if sig>0 else "🔴 SHORT"
                            msg = (
                                f"{side_tag} {denorm_sym(s)}\n"
                                f"TF: {tf}\n"
                                f"Entry≈ {entry:.6f}\n"
                                f"SL: {sl:.6f}   TP: {tp:.6f}\n"
                                f"Lev: {leverage}x   Qty≈ {qty:.6f}  Notional≈ {notional:.2f}\n"
                                f"Strat: {strat}"
                            )
                            if send_fills:
                                for cid in chat_ids:
                                    try: bot.send_message(chat_id=cid, text=msg)
                                    except Exception: pass
                            log("ORDER", msg.replace("\n"," | "))

                            last_sig[s] = sig

                except Exception as e:
                    es = f"{type(e).__name__}: {str(e)}"
                    log("ERROR", f"{denorm_sym(s)} -> {es}")
                    if send_fills:
                        try:
                            for cid in chat_ids:
                                try: bot.send_message(chat_id=cid, text=f"⚠️ Runtime error {denorm_sym(s)}\n{es[:200]}")
                                except Exception: pass
                        except Exception:
                            pass
                    if '429' in es or '418' in es or 'Too many requests' in es:
                        log("RATE", "Backoff 60s karena rate limit"); time.sleep(60)
            # ringkas tiap siklus
            log("TICK", f"processed={processed}/{len(symbols)}")

            # heartbeat telegram
            if hb_minutes > 0 and datetime.now(timezone.utc) >= next_hb:
                try:
                    valid_ts = [t for t in last_ts.values() if t is not None]
                    last_ts_str = (max(valid_ts).strftime("%Y-%m-%d %H:%M UTC") if valid_ts else "n/a")
                    hb = f"⏱️ Heartbeat: processed={processed}/{len(symbols)} | last_candle={last_ts_str}"
                    if send_fills:
                        for cid in chat_ids:
                            try: bot.send_message(chat_id=cid, text=hb)
                            except Exception: pass
                except Exception:
                    pass
                next_hb = datetime.now(timezone.utc) + timedelta(minutes=hb_minutes)

            time.sleep(poll_sleep)

        except KeyboardInterrupt:
            log("STOP", "KeyboardInterrupt")
            break
        except Exception as e:
            log("ERROR", f"main loop: {e}\n{traceback.format_exc()[:500]}")
            time.sleep(5)

    # shutdown ringkas
    try:
        if send_fills:
            for cid in chat_ids:
                try: bot.send_message(chat_id=cid, text="⏹️ LIVE Trader stopped")
                except Exception: pass
    except Exception:
        pass
    log("STOP", "Done")

if __name__ == "__main__":
    main()
