"""
╔══════════════════════════════════════════════════════════════════════╗
║   Strategy Name: ETH "Phantom Surge" Scalper                        ║
║   Timeframe: 30min                                                   ║
║   Symbol: ETH/USDT Perpetual Futures                                 ║
║   Type: Long + Short                                                 ╠
╠══════════════════════════════════════════════════════════════════════╣
║   FIX: Short PnL formula corrected.                                  ║
║   Formula: (entry - exit) / entry * 100                              ║
║   → positive when price fell (short profit)                          ║
║   → negative when price rose (short loss)                            ║
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
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

    # BB squeeze: width in bottom 25th percentile of last 50 candles
    df['bb_squeeze'] = df['bb_width'] < df['bb_width'].rolling(50).quantile(0.25)

    # BB expansion: width growing > 12% in last 2 candles
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

    # Swing highs/lows (last 10 candles lookback, excluding current)
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
    accel_up   = d['ema9_accel'] > 0.02
    accel_down = d['ema9_accel'] < -0.02

    # ── LAYER 5: Volume proof
    volume_spike = d['vol_ratio'] >= 1.8

    # ── LAYER 3: Strong candle
    strong_candle  = d['body_ratio'] >= 0.62
    body_expansion = d['body_rel']   >= 1.6

    # ══════════════════════════════════════
    #  LONG SIGNAL
    # ══════════════════════════════════════
    long_signal = all([
        squeeze_then_expand,
        close > d['swing_high'],                            # Layer 2: breakout above swing high
        strong_candle and (d['close'] > d['open']),         # Layer 3: bullish body
        accel_up and d['ema9_slope'] > 0.05,                # Layer 4: accelerating upside
        volume_spike,                                       # Layer 5: volume confirms
        body_expansion,                                     # urgency confirmation
    ])

    # ══════════════════════════════════════
    #  SHORT SIGNAL
    # ══════════════════════════════════════
    short_signal = all([
        squeeze_then_expand,
        close < d['swing_low'],                             # Layer 2: breakout below swing low
        strong_candle and (d['close'] < d['open']),         # Layer 3: bearish body
        accel_down and d['ema9_slope'] < -0.05,             # Layer 4: accelerating downside
        volume_spike,                                       # Layer 5: volume confirms
        body_expansion,
    ])

    if long_signal:
        sl  = close - 1.5 * atr
        tp1 = close + 2.0 * atr
        tp2 = close + 3.5 * atr
        return {
            "signal": "long",
            "entry": round(close, 4),
            "sl":    round(sl,  4),
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
            "sl":    round(sl,  4),
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
    idx:            int
    timestamp:      str
    signal:         str
    entry:          float
    sl:             float
    tp1:            float
    tp2:            float
    reason:         str
    exit_price:     float = 0.0
    exit_reason:    str   = ""
    pnl_pct:        float = 0.0
    candles_held:   int   = 0
    result:         str   = ""          # "win" | "loss" | "timeout"
    balance_before: float = 0.0
    position_size:  float = 0.0
    dollar_pnl:     float = 0.0
    balance_after:  float = 0.0


def backtest(df: pd.DataFrame, max_hold_candles: int = 8,
             starting_balance: float = 100.0,
             risk_pct: float = 0.10) -> list[Trade]:
    """
    starting_balance : initial USDT balance (default $100)
    risk_pct         : fraction of balance risked per trade (default 10%)

    PnL formula (unified for both directions):
      LONG:  (exit - entry) / entry * 100  → + when price rose
      SHORT: (entry - exit) / entry * 100  → + when price fell
    """
    results:   list[Trade]     = []
    in_trade:  bool            = False
    current_trade: Optional[Trade] = None
    balance = starting_balance

    for i in range(60, len(df)):
        row = df.iloc[i]

        # ── Manage open trade
        if in_trade and current_trade is not None:
            held = i - current_trade.idx
            hi   = row['high']
            lo   = row['low']
            cl   = row['close']

            if current_trade.signal == "long":
                if lo <= current_trade.sl:
                    current_trade.exit_price  = current_trade.sl
                    current_trade.exit_reason = "SL hit"
                    # Price fell to SL → loss
                    current_trade.pnl_pct     = (current_trade.sl - current_trade.entry) / current_trade.entry * 100
                    current_trade.result      = "loss"
                elif hi >= current_trade.tp1:
                    current_trade.exit_price  = current_trade.tp1
                    current_trade.exit_reason = "TP1 hit"
                    # Price rose to TP1 → win
                    current_trade.pnl_pct     = (current_trade.tp1 - current_trade.entry) / current_trade.entry * 100
                    current_trade.result      = "win"
                elif held >= max_hold_candles:
                    current_trade.exit_price  = cl
                    current_trade.exit_reason = "timeout"
                    current_trade.pnl_pct     = (cl - current_trade.entry) / current_trade.entry * 100
                    current_trade.result      = "win" if current_trade.pnl_pct > 0 else "loss"
                else:
                    current_trade.candles_held = held
                    continue

            elif current_trade.signal == "short":
                if hi >= current_trade.sl:
                    current_trade.exit_price  = current_trade.sl
                    current_trade.exit_reason = "SL hit"
                    # Price rose to SL → loss for short
                    current_trade.pnl_pct     = (current_trade.entry - current_trade.sl) / current_trade.entry * 100
                    current_trade.result      = "loss"
                elif lo <= current_trade.tp1:
                    current_trade.exit_price  = current_trade.tp1
                    current_trade.exit_reason = "TP1 hit"
                    # Price fell to TP1 → win for short
                    current_trade.pnl_pct     = (current_trade.entry - current_trade.tp1) / current_trade.entry * 100
                    current_trade.result      = "win"
                elif held >= max_hold_candles:
                    current_trade.exit_price  = cl
                    current_trade.exit_reason = "timeout"
                    # Short timeout: profit if price fell, loss if price rose
                    current_trade.pnl_pct     = (current_trade.entry - cl) / current_trade.entry * 100
                    current_trade.result      = "win" if current_trade.pnl_pct > 0 else "loss"
                else:
                    current_trade.candles_held = held
                    continue

            current_trade.candles_held  = held
            current_trade.dollar_pnl    = current_trade.position_size * current_trade.pnl_pct / 100
            current_trade.balance_after = current_trade.balance_before + current_trade.dollar_pnl
            balance = current_trade.balance_after
            results.append(current_trade)
            in_trade      = False
            current_trade = None

        # ── Look for new signal
        if not in_trade:
            sig = strategy(df.iloc[:i+1])
            if sig["signal"] != "none":
                current_trade = Trade(
                    idx            = i,
                    timestamp      = str(row['timestamp']),
                    signal         = sig["signal"],
                    entry          = sig["entry"],
                    sl             = sig["sl"],
                    tp1            = sig["tp1"],
                    tp2            = sig["tp2"],
                    reason         = sig["reason"],
                    balance_before = balance,
                    position_size  = balance * risk_pct,
                )
                in_trade = True

    return results


# ─────────────────────────────────────────────────────────────────────
# P&L CHART
# ─────────────────────────────────────────────────────────────────────

def save_pnl_chart(trades: list[Trade], starting_balance: float, out_path: str):
    """Saves a professional 4-panel P&L chart as PNG."""

    timestamps  = [pd.Timestamp(t.timestamp) for t in trades]
    balances    = [starting_balance] + [t.balance_after for t in trades]
    dollar_pnls = [t.dollar_pnl for t in trades]

    bal_arr  = np.array(balances[1:])
    peak_arr = np.maximum.accumulate(bal_arr)
    dd_arr   = (bal_arr - peak_arr) / peak_arr * 100

    BG      = "#0d1117"
    PANEL   = "#161b22"
    GREEN   = "#00e676"
    RED     = "#ff1744"
    YELLOW  = "#ffd600"
    BLUE    = "#448aff"
    GRAY    = "#30363d"
    TEXT    = "#e6edf3"
    SUBTEXT = "#8b949e"

    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        "ETH 'Phantom Surge' Scalper  —  P&L Dashboard  |  ETH/USDT 30m  |  2025",
        fontsize=15, fontweight="bold", color=TEXT, y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.96, top=0.93, bottom=0.07)

    # ── Panel 1: Balance Curve (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(PANEL)
    ax1.plot(timestamps, bal_arr, color=BLUE, linewidth=2, zorder=3)
    ax1.fill_between(timestamps, starting_balance, bal_arr, alpha=0.15, color=BLUE)
    ax1.axhline(starting_balance, color=GRAY, linestyle="--", linewidth=1, alpha=0.6)
    final_bal = balances[-1]
    ax1.annotate(f"  ${final_bal:.2f}", xy=(timestamps[-1], final_bal),
                 color=GREEN if final_bal > starting_balance else RED,
                 fontsize=11, fontweight="bold")
    ax1.set_title("Balance Curve (Starting $100 · 10% position size per trade)",
                  color=SUBTEXT, fontsize=10, pad=6)
    ax1.set_ylabel("Balance (USD)", color=TEXT, fontsize=9)
    ax1.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor(GRAY)

    # ── Panel 2: Trade-by-Trade Dollar P&L
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(PANEL)
    colors_bar = [GREEN if p >= 0 else RED for p in dollar_pnls]
    ax2.bar(range(len(dollar_pnls)), dollar_pnls, color=colors_bar, width=0.75, zorder=3)
    ax2.axhline(0, color=GRAY, linewidth=0.8)
    ax2.set_title("Trade P&L (USD)", color=SUBTEXT, fontsize=10, pad=6)
    ax2.set_xlabel("Trade #", color=SUBTEXT, fontsize=8)
    ax2.set_ylabel("$", color=TEXT, fontsize=9)
    ax2.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRAY)

    # ── Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(PANEL)
    ax3.fill_between(range(len(dd_arr)), dd_arr, 0, color=RED, alpha=0.35)
    ax3.plot(range(len(dd_arr)), dd_arr, color=RED, linewidth=1.2)
    ax3.set_title("Drawdown on Balance (%)", color=SUBTEXT, fontsize=10, pad=6)
    ax3.set_xlabel("Trade #", color=SUBTEXT, fontsize=8)
    ax3.set_ylabel("%", color=TEXT, fontsize=9)
    ax3.tick_params(colors=SUBTEXT, labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor(GRAY)

    # ── Panel 4: Monthly P&L
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor(PANEL)
    monthly: dict = {}
    for t in trades:
        m = pd.Timestamp(t.timestamp).strftime("%b")
        monthly[m] = monthly.get(m, 0) + t.dollar_pnl
    months  = list(monthly.keys())
    mpnls   = list(monthly.values())
    mcolors = [GREEN if v >= 0 else RED for v in mpnls]
    ax4.bar(months, mpnls, color=mcolors, zorder=3)
    ax4.axhline(0, color=GRAY, linewidth=0.8)
    ax4.set_title("Monthly P&L (USD)", color=SUBTEXT, fontsize=10, pad=6)
    ax4.set_ylabel("$", color=TEXT, fontsize=9)
    ax4.tick_params(colors=SUBTEXT, labelsize=8, axis='x', rotation=30)
    ax4.tick_params(colors=SUBTEXT, labelsize=8, axis='y')
    for spine in ax4.spines.values():
        spine.set_edgecolor(GRAY)

    # ── Panel 5: Stats summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(PANEL)
    ax5.axis("off")

    total   = len(trades)
    wins    = sum(1 for t in trades if t.result == "win")
    wr      = wins / total * 100 if total else 0
    net     = final_bal - starting_balance
    gross_w = sum(t.dollar_pnl for t in trades if t.dollar_pnl > 0)
    gross_l = sum(t.dollar_pnl for t in trades if t.dollar_pnl < 0)
    pf      = abs(gross_w / gross_l) if gross_l != 0 else float('inf')
    avg_w   = np.mean([t.dollar_pnl for t in trades if t.dollar_pnl > 0]) if wins else 0
    avg_l   = np.mean([t.dollar_pnl for t in trades if t.dollar_pnl < 0]) if total - wins else 0

    stats_lines = [
        ("Starting Balance", "$100.00"),
        ("Final Balance",    f"${final_bal:.2f}"),
        ("Net Profit",       f"${net:+.2f}  ({net/starting_balance*100:+.1f}%)"),
        ("Total Trades",     f"{total}"),
        ("Win Rate",         f"{wr:.1f}%"),
        ("Profit Factor",    f"{pf:.2f}"),
        ("Avg Win",          f"${avg_w:+.3f}"),
        ("Avg Loss",         f"${avg_l:+.3f}"),
        ("Max Drawdown",     f"{dd_arr.min():.2f}%"),
        ("Position Size",    "10% of balance per trade"),
    ]

    ax5.set_title("Summary Statistics", color=SUBTEXT, fontsize=10, pad=6)
    y = 0.95
    for label, value in stats_lines:
        val_color = TEXT
        if label == "Net Profit":
            val_color = GREEN if net >= 0 else RED
        elif label == "Win Rate":
            val_color = GREEN if wr >= 50 else YELLOW
        elif label == "Max Drawdown":
            val_color = RED
        ax5.text(0.02, y, label + ":", color=SUBTEXT, fontsize=9,
                 transform=ax5.transAxes, va="top")
        ax5.text(0.55, y, value, color=val_color, fontsize=9, fontweight="bold",
                 transform=ax5.transAxes, va="top")
        y -= 0.092

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  → Chart saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────

def print_report(trades: list[Trade], starting_balance: float = 100.0):
    if not trades:
        print("No trades generated.")
        return None

    total    = len(trades)
    wins     = [t for t in trades if t.result == "win"]
    losses   = [t for t in trades if t.result == "loss"]
    timeouts = [t for t in trades if t.result == "timeout"]
    win_rate = len(wins) / total * 100

    pnl_pcts    = [t.pnl_pct    for t in trades]
    dollar_pnls = [t.dollar_pnl for t in trades]
    net_dollar  = sum(dollar_pnls)
    final_bal   = trades[-1].balance_after

    avg_win    = np.mean([t.pnl_pct    for t in wins])   if wins   else 0
    avg_loss   = np.mean([t.pnl_pct    for t in losses]) if losses else 0
    avg_win_d  = np.mean([t.dollar_pnl for t in wins])   if wins   else 0
    avg_loss_d = np.mean([t.dollar_pnl for t in losses]) if losses else 0

    gross_w = sum(t.dollar_pnl for t in wins)
    gross_l = sum(t.dollar_pnl for t in losses)
    profit_factor = abs(gross_w / gross_l) if gross_l != 0 else float('inf')

    bal_arr  = np.array([t.balance_after for t in trades])
    peak_arr = np.maximum.accumulate(bal_arr)
    dd_arr   = (bal_arr - peak_arr) / peak_arr * 100
    max_dd   = dd_arr.min()

    longs    = [t for t in trades if t.signal == "long"]
    shorts   = [t for t in trades if t.signal == "short"]
    long_wr  = len([t for t in longs  if t.result == "win"]) / len(longs)  * 100 if longs  else 0
    short_wr = len([t for t in shorts if t.result == "win"]) / len(shorts) * 100 if shorts else 0
    avg_hold = np.mean([t.candles_held for t in trades])

    print("=" * 62)
    print("   🚀 ETH 'Phantom Surge' Scalper — Backtest Report")
    print("   📅 ETH/USDT 30m | 2025-01-01 → 2025-12-30")
    print("=" * 62)
    print(f"\n💵 ACCOUNT")
    print(f"   Starting balance : $100.00")
    print(f"   Final balance    : ${final_bal:.4f}")
    print(f"   Net profit       : ${net_dollar:+.4f}  ({net_dollar/starting_balance*100:+.2f}%)")
    print(f"   Position size    : 10% of balance per trade (compounding)")
    print(f"\n📊 TRADE SUMMARY")
    print(f"   Total trades     : {total}")
    print(f"   Wins             : {len(wins)}  ({win_rate:.1f}%)")
    print(f"   Losses           : {len(losses)}")
    print(f"   Timeouts         : {len(timeouts)}")
    print(f"\n💰 PnL METRICS")
    print(f"   Total PnL %      : {sum(pnl_pcts):+.2f}%")
    print(f"   Avg win          : {avg_win:+.2f}%  (${avg_win_d:+.4f})")
    print(f"   Avg loss         : {avg_loss:+.2f}%  (${avg_loss_d:+.4f})")
    print(f"   Profit factor    : {profit_factor:.2f}")
    print(f"   Max drawdown     : {max_dd:.2f}%")
    print(f"\n📈 LONG vs SHORT")
    print(f"   Longs   : {len(longs):3d} trades | WR {long_wr:.1f}%")
    print(f"   Shorts  : {len(shorts):3d} trades | WR {short_wr:.1f}%")
    print(f"\n⏱  AVG hold        : {avg_hold:.1f} candles ({avg_hold*0.5:.1f}h)")

    print(f"\n📅 MONTHLY BREAKDOWN")
    print(f"   {'Month':<10} {'Trades':>6} {'Win%':>6} {'PnL%':>8} {'$PnL':>8}")
    print(f"   {'-'*44}")
    for month_num in range(1, 13):
        m_trades = [t for t in trades if pd.Timestamp(t.timestamp).month == month_num]
        if not m_trades:
            continue
        m_wins  = len([t for t in m_trades if t.result == "win"])
        m_pnl   = sum(t.pnl_pct    for t in m_trades)
        m_dpnl  = sum(t.dollar_pnl for t in m_trades)
        m_wr    = m_wins / len(m_trades) * 100
        mname   = pd.Timestamp(f"2025-{month_num:02d}-01").strftime("%b")
        sign    = "+" if m_dpnl >= 0 else ""
        print(f"   {mname:<10} {len(m_trades):>6} {m_wr:>5.1f}% {m_pnl:>+7.2f}% {sign}{m_dpnl:>7.3f}$")

    top5 = sorted(trades, key=lambda t: t.dollar_pnl, reverse=True)[:5]
    print(f"\n🏆 TOP 5 TRADES (by $)")
    for t in top5:
        print(f"   {t.timestamp[:16]}  {t.signal.upper():<5}  "
              f"size=${t.position_size:.3f}  pnl={t.pnl_pct:+.2f}%  ${t.dollar_pnl:+.4f}  [{t.exit_reason}]")

    worst5 = sorted(trades, key=lambda t: t.dollar_pnl)[:5]
    print(f"\n💀 WORST 5 TRADES (by $)")
    for t in worst5:
        print(f"   {t.timestamp[:16]}  {t.signal.upper():<5}  "
              f"size=${t.position_size:.3f}  pnl={t.pnl_pct:+.2f}%  ${t.dollar_pnl:+.4f}  [{t.exit_reason}]")

    print("\n" + "=" * 62)
    print("⚠️  Known weaknesses: choppy sideways markets generate false")
    print("   squeezes; add a 1H EMA filter for live trading.")
    print("=" * 62)

    return {
        "total":            total,
        "win_rate":         win_rate,
        "net_dollar":       net_dollar,
        "final_balance":    final_bal,
        "profit_factor":    profit_factor,
        "max_drawdown":     max_dd,
        "avg_hold_candles": avg_hold,
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH    = "ETH_USDT_30m_20250101_20251230 - ETH_USDT_30m_20250101_20251230.csv.csv"
    CHART_PATH   = "./eth_pnl_chart.png"
    STARTING_BAL = 100.0
    RISK_PCT     = 0.10   # 10% of balance per trade

    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"  → {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    print("Computing indicators...")
    df = compute_indicators(df)

    print("Running backtest  (balance=$100 · pos=10% per trade)...")
    trades = backtest(df, max_hold_candles=8,
                      starting_balance=STARTING_BAL,
                      risk_pct=RISK_PCT)

    stats = print_report(trades, starting_balance=STARTING_BAL)

    print("\nGenerating P&L chart...")
    save_pnl_chart(trades, STARTING_BAL, CHART_PATH)
    print("Done. ✅")
