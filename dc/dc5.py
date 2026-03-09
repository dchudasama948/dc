"""
╔══════════════════════════════════════════════════════════════════════╗
║   Strategy Name: ETH "Phantom Surge" Scalper                        ║
║   Timeframe: 30min                                                   ║
║   Symbol: ETH/USDT Perpetual Futures                                 ║
║   Type: Long + Short                                                 ║
║   Author: Claude (crypto-trading-bot skill)                          ║
╚══════════════════════════════════════════════════════════════════════╝

STRATEGY LOGIC:
  All 5 layers must align simultaneously — no partial signals.

  Layer 1 — Volatility Phase:  BB squeeze → expansion (market loading)
  Layer 2 — Price Structure:   Price breaks above/below recent swing high/low
  Layer 3 — Candle Behavior:   Strong body (>65%) on breakout candle, not wick
  Layer 4 — Momentum:          EMA9 slope accelerating (2nd derivative positive)
  Layer 5 — Volume Proof:      Volume spike ≥ 1.8x 20-period average on signal candle

TARGET: 0.6% TP1, 1.2% TP2 | SL: ATR-based ~0.5% | RR: 1:1.2 – 1:2.4
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA 9, 21, 50
    df['ema9']  = df['close'].ewm(span=9,  adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

    # EMA9 slope (% change over 3 candles) — 1st derivative
    df['ema9_slope'] = (df['ema9'] - df['ema9'].shift(3)) / df['ema9'].shift(3) * 100

    # EMA9 slope acceleration — 2nd derivative (slope of slope)
    df['ema9_accel'] = df['ema9_slope'] - df['ema9_slope'].shift(2)

    # Bollinger Bands (20, 2)
    df['bb_mid']   = df['close'].rolling(20).mean()
    bb_std         = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # BB squeeze: width in bottom 20th percentile of last 50 candles
    df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)

    # BB expansion: width growing > 15% in last 2 candles
    df['bb_expand'] = (df['bb_width'] / df['bb_width'].shift(2) - 1) > 0.12

    # ATR(14)
    df['tr'] = np.maximum(df['high'] - df['low'],
               np.maximum(abs(df['high'] - df['close'].shift(1)),
                          abs(df['low']  - df['close'].shift(1))))
    df['atr14'] = df['tr'].rolling(14).mean()

    # Candle body ratio
    df['body']       = abs(df['close'] - df['open'])
    df['candle_rng'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['candle_rng'].replace(0, np.nan)

    # Body relative size vs prior 3 candles average
    df['avg_body3'] = df['body'].shift(1).rolling(3).mean()
    df['body_rel']  = df['body'] / df['avg_body3'].replace(0, np.nan)

    # Volume spike vs 20-period avg
    df['vol_avg20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_avg20'].replace(0, np.nan)

    # Swing highs/lows (last 10 candles lookback)
    df['swing_high'] = df['high'].shift(1).rolling(10).max()
    df['swing_low']  = df['low'].shift(1).rolling(10).min()

    # Hour of day for session filter
    df['hour'] = df['timestamp'].dt.hour

    return df


# ─────────────────────────────────────────────────────────────────────
# STRATEGY SIGNAL FUNCTION
# ─────────────────────────────────────────────────────────────────────

def strategy(df: pd.DataFrame) -> dict:
    """
    Given a DataFrame of candles (most recent = last row),
    return signal dict for the current bar close.
    """
    if len(df) < 60:
        return {"signal": "none", "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "reason": "insufficient data"}

    d = df.iloc[-1]  # current candle
    p = df.iloc[-2]  # previous candle

    close = d['close']
    atr   = d['atr14']

    if pd.isna(atr) or atr == 0:
        return {"signal": "none", "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "reason": "ATR not ready"}

    # ── SESSION FILTER: skip 00:00–03:00 UTC (low liquidity dead zone)
    if d['hour'] in [0, 1, 2, 3]:
        return {"signal": "none", "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "reason": "dead session"}

    # ── LAYER 1: Volatility state — squeeze then expansion
    squeeze_then_expand = bool(df['bb_squeeze'].iloc[-4:-2].any()) and bool(d['bb_expand'])

    # ── LAYER 4: Momentum — EMA9 slope accelerating
    accel_up   = d['ema9_accel'] > 0.02    # slope speeding upward
    accel_down = d['ema9_accel'] < -0.02   # slope speeding downward

    # ── LAYER 5: Volume proof
    volume_spike = d['vol_ratio'] >= 1.8

    # ── LAYER 3: Strong candle (body dominates range)
    strong_candle  = d['body_ratio'] >= 0.62
    body_expansion = d['body_rel'] >= 1.6   # this candle's body 60%+ larger than recent average

    # ══════════════════════════════════════
    #  LONG SIGNAL
    # ══════════════════════════════════════
    long_l2 = close > d['swing_high']         # Layer 2: price broke above swing high
    long_l3 = strong_candle and (d['close'] > d['open'])   # bullish body
    long_l4 = accel_up and d['ema9_slope'] > 0.05
    long_l5 = volume_spike
    long_l6 = d['ema9'] > d['ema21']          # trend alignment bonus

    long_signal = all([
        squeeze_then_expand,   # Layer 1
        long_l2,               # Layer 2: breakout
        long_l3,               # Layer 3: strong bull candle
        long_l4,               # Layer 4: accelerating upside
        long_l5,               # Layer 5: volume confirms
        body_expansion,        # size confirms urgency
    ])

    # ══════════════════════════════════════
    #  SHORT SIGNAL
    # ══════════════════════════════════════
    short_l2 = close < d['swing_low']
    short_l3 = strong_candle and (d['close'] < d['open'])   # bearish body
    short_l4 = accel_down and d['ema9_slope'] < -0.05
    short_l5 = volume_spike

    short_signal = all([
        squeeze_then_expand,
        short_l2,
        short_l3,
        short_l4,
        short_l5,
        body_expansion,
    ])

    if long_signal:
        sl  = close - 1.5 * atr
        tp1 = close + 2.0 * atr   # RR ~1:1.33
        tp2 = close + 3.5 * atr   # RR ~1:2.33
        return {
            "signal": "long",
            "entry": round(close, 4),
            "sl":    round(sl, 4),
            "tp1":   round(tp1, 4),
            "tp2":   round(tp2, 4),
            "reason": f"Phantom Surge LONG | BB expand:{d['bb_expand']:.0f} vol:{d['vol_ratio']:.2f}x accel:{d['ema9_accel']:.3f}"
        }

    if short_signal:
        sl  = close + 1.5 * atr
        tp1 = close - 2.0 * atr
        tp2 = close - 3.5 * atr
        return {
            "signal": "short",
            "entry": round(close, 4),
            "sl":    round(sl, 4),
            "tp1":   round(tp1, 4),
            "tp2":   round(tp2, 4),
            "reason": f"Phantom Surge SHORT | BB expand:{d['bb_expand']:.0f} vol:{d['vol_ratio']:.2f}x accel:{d['ema9_accel']:.3f}"
        }

    return {"signal": "none", "entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "reason": "no setup"}


# ─────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    idx: int
    timestamp: str
    signal: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    reason: str
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    candles_held: int = 0
    result: str = ""   # "win" | "loss" | "timeout"


def backtest(df: pd.DataFrame, max_hold_candles: int = 8) -> list[Trade]:
    results: list[Trade] = []
    in_trade = False
    current_trade: Optional[Trade] = None

    for i in range(60, len(df)):
        row = df.iloc[i]

        # ── If in a trade, check exit conditions
        if in_trade and current_trade is not None:
            held = i - current_trade.idx
            hi   = row['high']
            lo   = row['low']
            cl   = row['close']

            if current_trade.signal == "long":
                if lo <= current_trade.sl:
                    current_trade.exit_price  = current_trade.sl
                    current_trade.exit_reason = "SL hit"
                    current_trade.pnl_pct     = (current_trade.sl / current_trade.entry - 1) * 100
                    current_trade.result      = "loss"
                elif hi >= current_trade.tp1:
                    current_trade.exit_price  = current_trade.tp1
                    current_trade.exit_reason = "TP1 hit"
                    current_trade.pnl_pct     = (current_trade.tp1 / current_trade.entry - 1) * 100
                    current_trade.result      = "win"
                elif held >= max_hold_candles:
                    current_trade.exit_price  = cl
                    current_trade.exit_reason = "timeout"
                    current_trade.pnl_pct     = (cl / current_trade.entry - 1) * 100
                    current_trade.result      = "win" if current_trade.pnl_pct > 0 else "loss"
                else:
                    current_trade.candles_held = held
                    continue

            elif current_trade.signal == "short":
                if hi >= current_trade.sl:
                    current_trade.exit_price  = current_trade.sl
                    current_trade.exit_reason = "SL hit"
                    current_trade.pnl_pct     = (current_trade.entry / current_trade.sl - 1) * 100 * -1
                    current_trade.result      = "loss"
                elif lo <= current_trade.tp1:
                    current_trade.exit_price  = current_trade.tp1
                    current_trade.exit_reason = "TP1 hit"
                    current_trade.pnl_pct     = (current_trade.entry / current_trade.tp1 - 1) * 100
                    current_trade.result      = "win"
                elif held >= max_hold_candles:
                    current_trade.exit_price  = cl
                    current_trade.exit_reason = "timeout"
                    current_trade.pnl_pct     = (current_trade.entry / cl - 1) * 100
                    current_trade.result      = "win" if current_trade.pnl_pct > 0 else "loss"
                else:
                    current_trade.candles_held = held
                    continue

            current_trade.candles_held = held
            results.append(current_trade)
            in_trade      = False
            current_trade = None

        # ── Look for new signal (only if not in trade)
        if not in_trade:
            sig = strategy(df.iloc[:i+1])
            if sig["signal"] != "none":
                current_trade = Trade(
                    idx       = i,
                    timestamp = str(row['timestamp']),
                    signal    = sig["signal"],
                    entry     = sig["entry"],
                    sl        = sig["sl"],
                    tp1       = sig["tp1"],
                    tp2       = sig["tp2"],
                    reason    = sig["reason"],
                )
                in_trade = True

    return results


# ─────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────

def print_report(trades: list[Trade], df: pd.DataFrame):
    if not trades:
        print("No trades generated.")
        return

    total   = len(trades)
    wins    = [t for t in trades if t.result == "win"]
    losses  = [t for t in trades if t.result == "loss"]
    timeouts= [t for t in trades if t.result == "timeout"]
    win_rate= len(wins) / total * 100

    pnls    = [t.pnl_pct for t in trades]
    total_pnl = sum(pnls)
    avg_win   = np.mean([t.pnl_pct for t in wins])   if wins   else 0
    avg_loss  = np.mean([t.pnl_pct for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl_pct for t in wins) / sum(t.pnl_pct for t in losses)) if losses else float('inf')

    # Equity curve for max drawdown
    equity    = np.cumsum([1 + p/100 for p in pnls])
    peak      = np.maximum.accumulate(equity)
    drawdown  = (equity - peak) / peak * 100
    max_dd    = drawdown.min()

    # Longs vs Shorts
    longs  = [t for t in trades if t.signal == "long"]
    shorts = [t for t in trades if t.signal == "short"]
    long_wr  = len([t for t in longs  if t.result == "win"]) / len(longs)  * 100 if longs  else 0
    short_wr = len([t for t in shorts if t.result == "win"]) / len(shorts) * 100 if shorts else 0

    avg_hold = np.mean([t.candles_held for t in trades])

    print("=" * 60)
    print("   🚀 ETH 'Phantom Surge' Scalper — Backtest Report")
    print("   📅 ETH/USDT 30m | 2025-01-01 → 2025-12-30")
    print("=" * 60)
    print(f"\n📊 TRADE SUMMARY")
    print(f"   Total trades   : {total}")
    print(f"   Wins           : {len(wins)}  ({win_rate:.1f}%)")
    print(f"   Losses         : {len(losses)}")
    print(f"   Timeouts       : {len(timeouts)}")
    print(f"\n💰 PnL METRICS")
    print(f"   Total PnL      : {total_pnl:+.2f}%")
    print(f"   Avg win        : {avg_win:+.2f}%")
    print(f"   Avg loss       : {avg_loss:+.2f}%")
    print(f"   Profit factor  : {profit_factor:.2f}")
    print(f"   Max drawdown   : {max_dd:.2f}%")
    print(f"\n📈 LONG vs SHORT")
    print(f"   Longs   : {len(longs):3d} trades | WR {long_wr:.1f}%")
    print(f"   Shorts  : {len(shorts):3d} trades | WR {short_wr:.1f}%")
    print(f"\n⏱  AVG hold      : {avg_hold:.1f} candles ({avg_hold*0.5:.1f}h)")

    # Monthly breakdown
    print(f"\n📅 MONTHLY BREAKDOWN")
    print(f"   {'Month':<12} {'Trades':>6} {'Win%':>6} {'PnL%':>8}")
    print(f"   {'-'*36}")
    for month_num in range(1, 13):
        m_trades = [t for t in trades if pd.Timestamp(t.timestamp).month == month_num]
        if not m_trades:
            continue
        m_wins   = len([t for t in m_trades if t.result == "win"])
        m_pnl    = sum(t.pnl_pct for t in m_trades)
        m_wr     = m_wins / len(m_trades) * 100
        month_name = pd.Timestamp(f"2025-{month_num:02d}-01").strftime("%b")
        bar = "█" * int(abs(m_pnl) / 2)
        sign = "+" if m_pnl > 0 else ""
        print(f"   {month_name:<12} {len(m_trades):>6} {m_wr:>5.1f}% {sign}{m_pnl:>7.2f}%  {bar}")

    # Top 5 trades
    top5 = sorted(trades, key=lambda t: t.pnl_pct, reverse=True)[:5]
    print(f"\n🏆 TOP 5 TRADES")
    for t in top5:
        print(f"   {t.timestamp[:16]}  {t.signal.upper():<5}  entry={t.entry:.2f}  pnl={t.pnl_pct:+.2f}%  [{t.exit_reason}]")

    worst5 = sorted(trades, key=lambda t: t.pnl_pct)[:5]
    print(f"\n💀 WORST 5 TRADES")
    for t in worst5:
        print(f"   {t.timestamp[:16]}  {t.signal.upper():<5}  entry={t.entry:.2f}  pnl={t.pnl_pct:+.2f}%  [{t.exit_reason}]")

    print("\n" + "=" * 60)
    print("⚠️  Known weaknesses: choppy sideways markets generate false")
    print("   squeezes; add a 1H EMA filter for live trading.")
    print("=" * 60)

    return {
        "total": total,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "avg_hold_candles": avg_hold,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "ETH_USDT_30m_20250101_20251230 - ETH_USDT_30m_20250101_20251230.csv.csv"

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  → {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    print("Computing indicators...")
    df = compute_indicators(df)

    print("Running backtest...")
    trades = backtest(df, max_hold_candles=8)

    stats = print_report(trades, df)
