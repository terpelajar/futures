# live_trader_umf.py
from __future__ import annotations

import time, math, yaml, asyncio
from datetime import datetime, timezone, timedelta

import pandas as pd
from telegram import Bot
from binance.um_futures import UMFutures

from strategies import generate_signals, _atr
from universe import build_universe
from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 28  # aman untuk angka besar/kecil

# ---------- utils ----------
def now_str() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def log(level: str, msg: str):
    print(f"[{now_str()}] {level:<6} | {msg}", flush=True)


def sfloat(x, d=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(d)


def sint(x, d=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(d)


def last_valid_float(series: pd.Series, lookback: int = 5) -> float | None:
    s = pd.to_numeric(series.tail(lookback), errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else None


def norm_symbol(s: str) -> str:
    # "BTC/USDT" or "BTC-USDT" -> "BTCUSDT"
    return s.replace("/", "").replace("-", "").upper()


# ---------- telegram helper (handle async Bot) ----------
def tg_send(bot: Bot, chat_id: int, text: str):
    """Safe sender for python-telegram-bot v20+ coroutines from sync code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(bot.send_message(chat_id=chat_id, text=text))
    else:
        loop.create_task(bot.send_message(chat_id=chat_id, text=text))


# ---------- exchange ----------
def make_client(cfg: dict) -> UMFutures:
    live = cfg.get("live", {}) or {}
    keys = (cfg.get("keys", {}) or {}).get("binance", {})  # keys.binance
    api, sec = keys.get("api_key", ""), keys.get("secret", "")
    testnet = bool(live.get("testnet", True))
    base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    # sync client
    return UMFutures(key=api, secret=sec, base_url=base_url)


# ---------- market metadata ----------
def load_exchange_info(client: UMFutures) -> dict:
    return client.exchange_info()


def filter_usdtm_symbols(exch_info: dict, wanted: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    want = [norm_symbol(s) for s in wanted]
    ok, dropped = [], []
    by_symbol = {s["symbol"]: s for s in exch_info.get("symbols", [])}
    for w in want:
        m = by_symbol.get(w)
        if not m:
            dropped.append((w, "not listed"))
            continue
        # Syarat: futures & quote USDT; prefer PERPETUAL
        if m.get("status") != "TRADING":
            dropped.append((w, "status!=TRADING"))
            continue
        if m.get("quoteAsset") != "USDT":
            dropped.append((w, "quote!=USDT"))
            continue
        if m.get("contractType") not in ("PERPETUAL",):  # fokus UM perpetual
            dropped.append((w, "not PERPETUAL"))
            continue
        ok.append(w)
    return ok, dropped


def lot_step(symbol_info: dict) -> tuple[float | None, float | None]:
    step = None
    min_qty = None
    for f in symbol_info.get("filters", []):
        if f.get("filterType") in ("LOT_SIZE", "MARKET_LOT_SIZE"):
            step = sfloat(f.get("stepSize"), None)
            min_qty = sfloat(f.get("minQty"), None)
    return step, min_qty


def quantize_qty(sym: str, qty: float, exch_info: dict) -> float:
    info = next((s for s in exch_info.get("symbols", []) if s.get("symbol") == sym), None)
    if not info:  # fallback 6 desimal
        return math.floor(qty * 1e6) / 1e6
    step, min_qty = lot_step(info)
    if step and step > 0:
        qty = math.floor(qty / step) * step
    if min_qty and qty < min_qty:
        qty = 0.0
    return float(qty)
    
# ======== PRECISION HELPERS (tambahkan setelah quantize_qty) ========
def _dec(x) -> Decimal:
    return Decimal(str(x))

def _step_decimals(step) -> int:
    s = str(step)
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0

def _symbol_info(exch_info: dict, sym: str) -> dict | None:
    return next((s for s in exch_info.get("symbols", []) if s.get("symbol") == sym), None)

def _tick_size(info: dict) -> float | None:
    for f in info.get("filters", []):
        if f.get("filterType") == "PRICE_FILTER":
            ts = f.get("tickSize")
            return float(ts) if ts is not None else None
    return None

def _qty_step(info: dict) -> float | None:
    # MARKET order → prefer MARKET_LOT_SIZE; fallback LOT_SIZE
    mls = None
    ls  = None
    for f in info.get("filters", []):
        if f.get("filterType") == "MARKET_LOT_SIZE":
            mls = f.get("stepSize")
        elif f.get("filterType") == "LOT_SIZE":
            ls = f.get("stepSize")
    step = mls if mls is not None else ls
    return float(step) if step is not None else None

def quantize_price_str(sym: str, price: float, exch_info: dict) -> str:
    """Bulatkan harga ke tickSize dan format string dengan jumlah desimal yang benar."""
    info = _symbol_info(exch_info, sym)
    if not info:
        return f"{price:.6f}"
    tick = _tick_size(info) or 0.01
    d = _step_decimals(tick)
    step_d = _dec(tick)
    q = (_dec(price) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return f"{q:.{d}f}"

def quantize_qty_str(sym: str, qty: float, exch_info: dict) -> str:
    """Bulatkan qty ke stepSize dan format string dengan jumlah desimal yang benar."""
    info = _symbol_info(exch_info, sym)
    if not info:
        q = (_dec(qty) * _dec("1e6")).to_integral_value(rounding=ROUND_DOWN) / _dec("1e6")
        return f"{q:.6f}"
    step = _qty_step(info) or 0.000001
    d = _step_decimals(step)
    step_d = _dec(step)
    q = (_dec(qty) / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    return f"{q:.{d}f}"
# ================================================================



# ---------- sizing ----------
def qty_from_risk(entry: float, sl: float, equity_usdt: float, risk_per_trade: float) -> tuple[float, float]:
    stop_abs = abs(entry - sl)
    risk_amount = equity_usdt * risk_per_trade
    if stop_abs <= 0 or risk_amount <= 0:
        return 0.0, 0.0
    return risk_amount / stop_abs, risk_amount


def compute_levels(df: pd.DataFrame, side: int, params: dict, risk: dict, entry_source: str = "close"):
    price = df["open"] if entry_source == "open" else df["close"]
    entry = last_valid_float(price, 5)
    if entry is None:
        raise ValueError("no valid entry")

    rr = sfloat(risk.get("take_profit_rr", 2.0), 2.0)
    stop_mode = (risk.get("stop_mode") or "atr").lower()

    if stop_mode == "atr":
        ap = sint(params.get("atr_entry_period", 14), 14)
        am = sfloat(params.get("atr_entry_mult", 1.5), 1.5)
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
        slp = sfloat(risk.get("stop_loss_pct", 0.02), 0.02)
        tpp = risk.get("take_profit_pct")
        tpp = sfloat(tpp if tpp is not None else rr * slp, rr * slp)
        sl = entry * (1 - slp) if side > 0 else entry * (1 + slp)
        tp = entry * (1 + tpp) if side > 0 else entry * (1 - tpp)

    # minimal SL (opsional)
    min_sl = sfloat(risk.get("min_sl_pct", 0.0), 0.0)
    if min_sl > 0.0:
        cur_frac = abs((sl / entry - 1.0)) if side > 0 else abs((1.0 - sl / entry))
        if cur_frac < min_sl:
            if side > 0:
                sl = entry * (1 - min_sl)
                tp = entry + rr * (entry - sl)
            else:
                sl = entry * (1 + min_sl)
                tp = entry - rr * (sl - entry)

    return float(entry), float(sl), float(tp)


# ---------- account helpers ----------
def futures_free_usdt(client: UMFutures) -> float:
    # /fapi/v2/balance
    try:
        data = client.balance()
        # list of dict: [{'asset':'USDT', 'availableBalance':'...'}, ...]
        for a in data:
            if a.get("asset") == "USDT":
                free = a.get("availableBalance") or a.get("balance") or "0"
                return max(float(free), 0.0)
    except Exception as e:
        log("WARN", f"balance: {e}")
    return 0.0


def leverage_bracket_cap(client: UMFutures, symbol: str, leverage: int) -> float:
    """Ambil notionalCap yang cocok untuk leverage. Kompatibel beberapa versi client."""
    try:
        # Prefer API modern: leverage_brackets (jamak)
        if hasattr(client, "leverage_brackets"):
            data = client.leverage_brackets(symbol=symbol)
        else:
            # Fallback: beberapa versi expose singular
            data = client.leverage_bracket(symbol=symbol)

        if isinstance(data, dict):
            brackets = data.get("brackets", [])
        elif isinstance(data, list):
            item = next((it for it in data if it.get("symbol") == symbol), data[0] if data else {})
            brackets = item.get("brackets", [])
        else:
            brackets = []

        caps_ok, caps_all = [], []
        for b in brackets:
            ilv = int(b.get("initialLeverage") or 0)
            cap = float(b.get("notionalCap") or 0.0)
            caps_all.append(cap)
            if ilv >= int(leverage):
                caps_ok.append(cap)

        if caps_ok:
            return max(caps_ok)
        if caps_all:
            return min(caps_all)
        return float("inf")
    except Exception as e:
        log("WARN", f"leverage_bracket {symbol}: {e}")
        return float("inf")


def clamp_by_free_and_bracket(
    client: UMFutures,
    exch_info: dict,
    symbol: str,
    qty_init: float,
    entry: float,
    leverage: int,
    margin_safety: float = 0.95,
    bracket_safety: float = 0.98,
) -> tuple[float, dict]:
    info = {}
    notional_init = qty_init * entry
    free = futures_free_usdt(client)
    cap = leverage_bracket_cap(client, symbol, leverage)

    allowed_free_notional = free * leverage * margin_safety
    allowed_bracket_notional = (cap * bracket_safety) if cap != float("inf") else float("inf")
    allowed_notional = min(notional_init, allowed_free_notional, allowed_bracket_notional)

    info.update(
        dict(
            free_usdt=free,
            cap_notional=cap,
            notional_init=notional_init,
            allowed_free_notional=allowed_free_notional,
            allowed_bracket_notional=allowed_bracket_notional,
            allowed_notional=allowed_notional,
        )
    )

    if allowed_notional <= 0:
        return 0.0, info

    qty_target = allowed_notional / max(entry, 1e-12)
    qty_final = quantize_qty(symbol, qty_target, exch_info)
    return float(qty_final), info


# ---------- futures account settings ----------
def ensure_settings(client: UMFutures, exch_info: dict, symbol: str, leverage: int, margin_type: str, position_mode: str):
    # position mode
    try:
        dual = (position_mode or "").lower() == "hedge"
        client.change_position_mode(dualSidePosition="true" if dual else "false")
    except Exception:
        pass
    # margin type & leverage
    try:
        client.change_margin_type(symbol=symbol, marginType=margin_type.upper())
    except Exception:
        pass
    try:
        client.change_leverage(symbol=symbol, leverage=int(leverage))
    except Exception:
        pass


# ---------- orders ----------
def cancel_all_open_orders(client: UMFutures, symbol: str):
    try:
        client.cancel_open_orders(symbol=symbol)
    except Exception as e:
        log("WARN", f"cancel_open_orders {symbol}: {e}")


def place_protective_orders(client: UMFutures, symbol: str, side: int, sl: float, tp: float,
                            working_type: str, position_mode: str, exch_info: dict):
    """
    Pasang SL/TP close-all:
      - type: STOP_MARKET / TAKE_PROFIT_MARKET
      - gunakan closePosition=True
      - JANGAN kirim reduceOnly atau quantity (akan error -1106).
    """
    base = dict(timeInForce="GTC", workingType=working_type, closePosition=True)
    if (position_mode or "").lower() == "hedge":
        base["positionSide"] = "LONG" if side > 0 else "SHORT"

    # DEFINISIKAN sisi order untuk SL/TP
    sl_side = "SELL" if side > 0 else "BUY"
    tp_side = "SELL" if side > 0 else "BUY"

    # Quantize harga ke tickSize
    sl_s = quantize_price_str(symbol, sl, exch_info)
    tp_s = quantize_price_str(symbol, tp, exch_info)

    try:
        client.new_order(symbol=symbol, side=sl_side, type="STOP_MARKET", stopPrice=sl_s, **base)
    except Exception as e:
        log("ERROR", f"place SL {symbol}: {e}")
    try:
        client.new_order(symbol=symbol, side=tp_side, type="TAKE_PROFIT_MARKET", stopPrice=tp_s, **base)
    except Exception as e:
        log("ERROR", f"place TP {symbol}: {e}")



# ---------- klines ----------
def fetch_df(client: UMFutures, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    # kline format: [openTime, open, high, low, close, volume, closeTime, ...]
    raw = client.klines(symbol=symbol, interval=interval, limit=limit)
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    cols = ["openTime", "open", "high", "low", "close", "volume", "closeTime", "q", "n", "tbb", "tbq", "ig"]
    df = pd.DataFrame(raw, columns=cols[: len(raw[0])])
    df["timestamp"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume"]].dropna()


# ---------- one-shot snapshot ----------
def send_snapshot(
    bot: Bot,
    chat_ids,
    client: UMFutures,
    symbols: list[str],
    tf: str,
    lookback: int,
    default_strategy: str,
    strategy_map: dict,
    base_params: dict,
    params_over: dict,
    risk: dict,
    exch_info: dict,
):
    lines = []
    for sym in symbols:
        try:
            df = fetch_df(client, sym, tf, min(lookback, 1000))
            if len(df) < 60:
                lines.append(f"{sym}: data<60 bars")
                continue
            strat = strategy_map.get(sym, default_strategy)
            params = dict(base_params)
            params.update(params_over.get(sym, {}))
            sig_df = generate_signals(df, strat, params)
            sig = int(sig_df["signal"].iloc[-1])
            entry, sl, tp = compute_levels(df, sig if sig != 0 else 1, params, risk, entry_source=str(risk.get("entry_source", "close")))
            tp_pct = (tp / entry - 1.0) * 100.0 if sig > 0 else (1.0 - tp / entry) * 100.0 if sig < 0 else 0.0
            lines.append(f"{sym}: sig={sig}  TP≈{abs(tp_pct):.2f}%")
        except Exception as e:
            lines.append(f"{sym}: err {str(e)[:60]}")
    msg = "📸 Snapshot (latest signals)\n" + "\n".join(lines[:20])
    for cid in chat_ids:
        try:
            tg_send(bot, cid, msg)
        except Exception:
            pass


# ---------- main ----------
def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exch = cfg.get("exchange", "binance")
    market_type = cfg.get("market_type", "futures")
    if (exch or "").lower() != "binance" or (market_type or "").lower() not in ("futures", "future"):
        raise RuntimeError("Script ini khusus Binance USDⓈ-M Futures")

    tf = cfg["timeframe"]
    lookback = sint(cfg.get("lookback_candles", 300), 300)

    # universe input bisa "BTC/USDT" dst → normalkan ke "BTCUSDT"
    symbols_in = (
        build_universe(cfg)
        if cfg.get("universe", {}).get("mode") == "dynamic"
        else (cfg.get("universe", {}).get("symbols", []) or cfg.get("symbols", []))
    )
    if not symbols_in:
        raise RuntimeError("No symbols")
    symbols_req = [norm_symbol(s) for s in symbols_in]

    # telegram
    tg = cfg["telegram"]
    bot = Bot(token=tg["bot_token"])
    chat_ids = tg["chat_ids"]

    # risk & live
    risk = cfg.get("risk", {}) or {}
    leverage = sint(risk.get("leverage", 3), 3)
    rpt = sfloat(risk.get("risk_per_trade", 0.01), 0.01)
    max_margin_frac = sfloat(risk.get("max_margin_frac", 0.2), 0.2)  # not used now; keep for future caps
    entry_source = str(risk.get("entry_source", "close"))

    live = cfg.get("live", {}) or {}
    if not live.get("enabled", False):
        raise RuntimeError("live.enabled must be true")
    position_mode = live.get("position_mode", "oneway")
    margin_type = live.get("margin_type", "ISOLATED")
    working_type = live.get("working_type", "MARK_PRICE")
    send_fills = bool(live.get("send_fills_to_telegram", True))
    poll_sleep = sint(live.get("poll_sleep", 10), 10)
    hb_minutes = sint(live.get("heartbeat_minutes", 0), 0)

    # strategi
    default_strategy = (cfg.get("strategies") or ["ema_cross"])[0]
    strategy_map = cfg.get("strategy_per_symbol", {}) or {}
    base_params = cfg.get("params", {}) or {}
    params_over = cfg.get("params_overrides", {}) or {}

    # equity bayangan (buat sizing awal)
    equity_usdt = sfloat(cfg.get("account", {}).get("equity_usdt", 1000.0), 1000.0)

    client = make_client(cfg)
    exch_info = load_exchange_info(client)
    ok, dropped = filter_usdtm_symbols(exch_info, symbols_req)
    if dropped:
        log("FILTER", f"dropped {len(dropped)}: " + ", ".join([d[0] for d in dropped][:20]) + ("..." if len(dropped) > 20 else ""))
    if not ok:
        # fallback 5 besar
        fallback = [
            s
            for s in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"]
            if any(x.get("symbol") == s for x in exch_info.get("symbols", []))
        ]
        if not fallback:
            raise RuntimeError("No valid symbols after filter")
        ok = fallback
        log("FILTER", "using fallback: " + ", ".join(ok))
    symbols = ok

    # apply account settings per symbol
    for sym in symbols:
        ensure_settings(client, exch_info, sym, leverage, margin_type, position_mode)

    # startup
    start_text = (
        "🚀 LIVE Trader (UMFutures) started\n"
        f"Exchange: {exch} (futures {'TESTNET' if (live.get('testnet', True)) else 'LIVE'})\n"
        f"TF: {tf}  Symbols: {len(symbols)}  Leverage: {leverage}x  Mode: {position_mode}/{margin_type}\n"
        f"workingType={working_type}, risk_per_trade={rpt*100:.2f}%  equity≈{equity_usdt:.2f}"
    )
    for cid in chat_ids:
        try:
            tg_send(bot, cid, start_text)
        except Exception:
            pass
    log("START", start_text.replace("\n", " | "))

    # snapshot sekali
    send_snapshot(
        bot,
        chat_ids,
        client,
        symbols,
        tf,
        lookback,
        default_strategy,
        strategy_map,
        base_params,
        params_over,
        risk,
        exch_info,
    )

    last_ts = {s: None for s in symbols}
    last_sig = {s: None for s in symbols}
    next_hb = datetime.now(timezone.utc)

    try:
        while True:
            processed = 0
            for sym in symbols:
                try:
                    df = fetch_df(client, sym, tf, min(lookback, 1000))
                    processed += 1
                    if len(df) < 60:
                        continue

                    newest = df.index[-1]
                    if last_ts[sym] is None or newest > last_ts[sym]:
                        last_ts[sym] = newest

                        strat = strategy_map.get(sym, default_strategy)
                        params = dict(base_params)
                        params.update(params_over.get(sym, {}))
                        sig_df = generate_signals(df, strat, params)
                        sig = int(sig_df["signal"].iloc[-1])

                        if sig != 0 and last_sig.get(sym) != sig:
                            entry, sl, tp = compute_levels(df, sig, params, risk, entry_source=entry_source)

                            # sizing by risk (pakai equity_usdt config)
                            qty_raw, _ = qty_from_risk(entry, sl, equity_usdt, rpt)
                            qty_init = quantize_qty(sym, qty_raw, exch_info)
                            if qty_init <= 0:
                                log("SKIP", f"{sym} qty<=0 (risk)")
                                last_sig[sym] = sig
                                continue

                            # clamp by free & bracket (pakai saldo akun real)
                            qty, clamp = clamp_by_free_and_bracket(
                                client,
                                exch_info,
                                sym,
                                qty_init,
                                entry,
                                leverage,
                                margin_safety=0.95,
                                bracket_safety=0.98,
                            )
                            if qty <= 0:
                                log("SKIP", f"{sym} qty=0 clamp | free={clamp.get('free_usdt')} cap={clamp.get('cap_notional')}")
                                last_sig[sym] = sig
                                continue

                            notional = qty * entry
                            # (opsi tambahan: hard cap notional, jika diperlukan)
                            max_notional_cap = sfloat(risk.get("max_notional_usdt", 0.0), 0.0)
                            if max_notional_cap > 0 and notional > max_notional_cap:
                                scale = max_notional_cap / notional
                                qty = quantize_qty(sym, qty * scale, exch_info)
                                notional = qty * entry
                                if qty <= 0:
                                    log("SKIP", f"{sym} qty after cap<=0")
                                    last_sig[sym] = sig
                                    continue

                            # cancel open orders lama
                            cancel_all_open_orders(client, sym)

                            # MARKET entry
                            side_txt = "BUY" if sig > 0 else "SELL"
                            entry_params = {}
                            if (position_mode or "").lower() == "hedge":
                                entry_params["positionSide"] = "LONG" if sig > 0 else "SHORT"
                            try:
                                qty_s = quantize_qty_str(sym, qty, exch_info)
                                client.new_order(symbol=sym, side=side_txt, type="MARKET", quantity=qty_s, **entry_params)

                            except Exception as e:
                                log("ERROR", f"order {sym}: {e}")
                                last_sig[sym] = sig
                                continue

                            # SL/TP close-all (tanpa reduceOnly, tanpa quantity)
                            place_protective_orders(client, sym, sig, sl, tp, working_type, position_mode, exch_info)

                            # Telegram
                            msg = (
                                f"{'🟢 LONG' if sig > 0 else '🔴 SHORT'} {sym}\n"
                                f"TF: {tf}\n"
                                f"Entry≈ {entry:.6f}\n"
                                f"SL: {sl:.6f}   TP: {tp:.6f}\n"
                                f"Lev: {leverage}x   Qty≈ {qty:.6f}  Notional≈ {notional:.2f}\n"
                                f"Strat: {strat}"
                            )
                            if send_fills:
                                for cid in chat_ids:
                                    try:
                                        tg_send(bot, cid, msg)
                                    except Exception:
                                        pass
                            log("ORDER", msg.replace("\n", " | "))
                            last_sig[sym] = sig

                except Exception as e:
                    log("ERROR", f"{sym} -> {e}")

            log("TICK", f"processed={processed}/{len(symbols)}")

            # Heartbeat
            if hb_minutes > 0 and datetime.now(timezone.utc) >= next_hb:
                try:
                    valid = [t for t in last_ts.values() if t is not None]
                    last_candle = max(valid).strftime("%Y-%m-%d %H:%M UTC") if valid else "n/a"
                    hb = f"⏱️ Heartbeat: processed={processed}/{len(symbols)} | last_candle={last_candle}"
                    if send_fills:
                        for cid in chat_ids:
                            try:
                                tg_send(bot, cid, hb)
                            except Exception:
                                pass
                except Exception:
                    pass
                next_hb = datetime.now(timezone.utc) + timedelta(minutes=hb_minutes)

            time.sleep(poll_sleep)

    finally:
        try:
            for cid in chat_ids:
                try:
                    tg_send(bot, cid, "⏹️ LIVE Trader (UMF) stopped")
                except Exception:
                    pass
        except Exception:
            pass
        log("STOP", "Done")


if __name__ == "__main__":
    main()
