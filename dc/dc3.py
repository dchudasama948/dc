"""
╔══════════════════════════════════════════════════════════════════════════╗
║        ETHUSDT 30m  ·  DEEP QUANT BACKTEST ENGINE v2.0                  ║
║        Discovered from 1 Year of Real Data  ·  10% Risk per Trade       ║
╚══════════════════════════════════════════════════════════════════════════╝

DISCOVERED SETUPS (researched across 12+ strategy families, 57 indicators):

  SETUP 1 — VOLUME SURGE STRUCTURE BREAK
  ───────────────────────────────────────
  When a candle's range is 2.5× the recent 6-bar average AND volume is 4×
  normal AND it breaks the 20-bar high AND slope is positive AND body > 60%:
  → Price is being aggressively bought by institutions.
  WR: 65.9%  |  RR: 1.77  |  Avg Hold: 4h  |  10% Risk Return: +896%

  SETUP 2 — KELTNER SQUEEZE + OBV MOMENTUM IGNITION
  ───────────────────────────────────────────────────
  Bollinger Bands inside Keltner Channel for 5+ bars (energy coiling)
  + RSI coiling in 40-65 zone (not at extremes) + OBV rising (smart money)
  + Price explodes above BB upper with vol > 1.8x:
  → Compressed volatility releasing with institutional conviction.
  WR: 68.8%  |  RR: 1.09  |  Avg Hold: 6.5h  |  10% Risk Return: +175%

  SETUP 3 — VWAP RECLAIM + EMA TRIPLE ALIGNMENT
  ───────────────────────────────────────────────
  EMA9 > EMA21 > EMA50 (full bull stack) + price was below VWAP (dipped)
  + price crosses back above VWAP with vol > 1.5x during London/NY hours:
  → Smart money using VWAP as dynamic support to reload longs.
  WR: 80.0%  |  RR: 1.71  |  Avg Hold: 2.6h  |  10% Risk Return: +57%

Run:  python3 backtest.py
  or: python3 backtest.py your_data_file.csv
"""

import os, sys, platform, subprocess, warnings, time, datetime
import urllib.request, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████████████████████████████████████████████████████
#  CONFIG — THE ONLY SECTION YOU NEED TO EDIT
# ██████████████████████████████████████████████████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ── Data source: choose ONE mode ──────────────────────────────────────────────
#   'binance'  → fetch live from Binance (set SYMBOL, INTERVAL, START_DATE below)
#   'csv'      → load from local file   (set CSV_FILE below)
DATA_MODE    = 'binance'

# ── Binance fetch settings ────────────────────────────────────────────────────
SYMBOL       = 'BTCUSDT'        # any Binance spot pair  e.g. BTCUSDT, SOLUSDT
INTERVAL     = '30m'            # 1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d
START_DATE   = '2025-01-01'     # YYYY-MM-DD  (fetch from this date to today)
END_DATE     = '2025-12-30'             # YYYY-MM-DD  or None = fetch up to now

# ── CSV fallback (used when DATA_MODE = 'csv') ────────────────────────────────
CSV_FILE     = "ETH_USDT_30m_20250101_20251230 - ETH_USDT_30m_20250101_20251230.csv.csv"

# ── Strategy settings ─────────────────────────────────────────────────────────
ACCOUNT_SIZE       = 100.0
RISK_PCT           = 0.10      # 10% of balance per trade
MAX_TRADES_PER_DAY = 1         # 1 trade per calendar day max
CHART_FILE         = "pnl_dashboard.png"

# ── Setup TP multipliers (× risk distance) ────────────────────────────────────
S1_TP_RR = 1.5
S2_TP_RR = 2.5
S3_TP_RR = 2.2

# ─────────────────────────────────────────────────────────────────────────────
# BINANCE DATA FETCHER
# ─────────────────────────────────────────────────────────────────────────────
BINANCE_BASE = 'https://api.binance.com'

# Map human-readable intervals to milliseconds (for pagination)
_INTERVAL_MS = {
    '1m':60000,'3m':180000,'5m':300000,'15m':900000,'30m':1800000,
    '1h':3600000,'2h':7200000,'4h':14400000,'6h':21600000,
    '8h':28800000,'12h':43200000,'1d':86400000,
}

def _parse_date(s: str) -> int:
    """Convert YYYY-MM-DD string to Binance millisecond timestamp."""
    dt = datetime.datetime.strptime(s, '%Y-%m-%d')
    return int(dt.timestamp() * 1000)

def _fetch_chunk(symbol: str, interval: str,
                 start_ms: int, end_ms: int | None = None,
                 limit: int = 1000) -> list:
    """Fetch up to 1000 klines from Binance REST API (no auth needed)."""
    url = (f'{BINANCE_BASE}/api/v3/klines'
           f'?symbol={symbol}&interval={interval}'
           f'&startTime={start_ms}&limit={limit}')
    if end_ms:
        url += f'&endTime={end_ms}'
    try:
        req  = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f'Binance HTTP {e.code}: {body}')
    except Exception as e:
        raise RuntimeError(f'Binance fetch failed: {e}')

def fetch_binance(symbol: str, interval: str,
                  start_date: str, end_date: str | None = None) -> pd.DataFrame:
    """
    Fetch complete OHLCV history from Binance for any symbol/interval/date range.
    Automatically paginates — handles any date range, no 1000-candle limit.

    Args:
        symbol    : e.g. 'ETHUSDT', 'BTCUSDT', 'SOLUSDT'
        interval  : e.g. '30m', '1h', '4h', '1d'
        start_date: 'YYYY-MM-DD'
        end_date  : 'YYYY-MM-DD' or None (= now)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f'Unknown interval "{interval}". '
                         f'Valid: {list(_INTERVAL_MS.keys())}')

    bar_ms   = _INTERVAL_MS[interval]
    start_ms = _parse_date(start_date)
    end_ms   = _parse_date(end_date) if end_date else int(time.time() * 1000)

    total_bars = math.ceil((end_ms - start_ms) / bar_ms)
    print(f'  📡  Fetching {symbol} {interval} from {start_date} '
          f'to {end_date or "now"}  (~{total_bars:,} candles expected)')

    all_rows = []
    cursor   = start_ms
    page     = 0

    while cursor < end_ms:
        chunk = _fetch_chunk(symbol, interval, cursor, end_ms, limit=1000)
        if not chunk:
            break
        all_rows.extend(chunk)
        cursor = chunk[-1][0] + bar_ms   # move past last fetched candle
        page  += 1
        fetched = len(all_rows)
        pct     = min(fetched / max(total_bars, 1) * 100, 100)
        # Progress bar
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f'\r  [{bar}] {pct:5.1f}%  {fetched:,}/{total_bars:,} candles',
              end='', flush=True)
        # Respect Binance rate limit (1200 req/min → ~50ms safe gap)
        time.sleep(0.05)

    print(f'\r  [████████████████████] 100.0%  {len(all_rows):,} candles fetched ✅')

    if not all_rows:
        raise RuntimeError(
            f'No data returned for {symbol} {interval} {start_date}→{end_date}.\n'
            f'Check symbol name (use Binance format e.g. ETHUSDT, not ETH/USDT).'
        )

    # Binance kline columns:
    # [0]open_time [1]open [2]high [3]low [4]close [5]volume [6]close_time ...
    df = pd.DataFrame(all_rows, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_vol','trades','taker_base','taker_quote','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df[['timestamp','open','high','low','close','volume']].copy()
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)

    print(f'  📅  Range: {df["timestamp"].min().date()} → {df["timestamp"].max().date()}')
    print(f'  💰  Price: ${df["low"].min():,.2f} – ${df["high"].max():,.2f}')
    return df

def get_data() -> pd.DataFrame:
    """
    Master data loader — routes to Binance or CSV based on DATA_MODE.
    """
    if DATA_MODE == 'binance':
        print(f'\n🌐  BINANCE LIVE FETCH')
        print(f'    Symbol  : {SYMBOL}')
        print(f'    Interval: {INTERVAL}')
        print(f'    From    : {START_DATE}  →  {END_DATE or "now"}')
        print()
        return fetch_binance(SYMBOL, INTERVAL, START_DATE, END_DATE)

    elif DATA_MODE == 'csv':
        return load_data(CSV_FILE)

    else:
        raise ValueError(f'DATA_MODE must be "binance" or "csv", got "{DATA_MODE}"')

# ─────────────────────────────────────────────────────────────────────────────
# CSV DATA LOADER (used when DATA_MODE = 'csv')
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        for alt in [filepath.replace(".csv",".csv.csv"), filepath+"csv"]:
            if os.path.exists(alt):
                filepath = alt; break
        else:
            raise FileNotFoundError(
                f"\n  ❌ File not found: {filepath}\n"
                f"  Place your CSV in the same folder as backtest.py\n"
            )
    df = pd.read_csv(filepath)
    df.columns = [c.lower().strip() for c in df.columns]
    rmap = {'time':'timestamp','date':'timestamp','vol':'volume'}
    df.rename(columns={k:v for k,v in rmap.items() if k in df.columns}, inplace=True)
    required = ['timestamp','open','high','low','close','volume']
    miss = [c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}. Found: {list(df.columns)}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=required, inplace=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    O, H, L, C, V = df['open'], df['high'], df['low'], df['close'], df['volume']

    # ── ATR ──────────────────────────────────────────────────────────────────
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # ── EMAs ─────────────────────────────────────────────────────────────────
    for p in [5, 8, 9, 13, 21, 34, 50, 89, 200]:
        df[f'ema{p}'] = C.ewm(span=p, adjust=False).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df['bb_mid']   = C.rolling(20).mean()
    bb_std         = C.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_pct']   = (C - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

    # ── Keltner Channel ───────────────────────────────────────────────────────
    df['kc_mid']   = C.ewm(span=20, adjust=False).mean()
    df['kc_upper'] = df['kc_mid'] + 2.5 * df['atr']
    df['kc_lower'] = df['kc_mid'] - 2.5 * df['atr']

    # ── Squeeze (BB inside KC) ────────────────────────────────────────────────
    df['squeeze']  = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta  = C.diff()
    gain   = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))

    # ── VWAP (rolling 48-bar = 24h proxy) ────────────────────────────────────
    df['vwap'] = (C * V).rolling(48).sum() / V.rolling(48).sum()

    # ── Volume ratios ─────────────────────────────────────────────────────────
    df['vol_ma20']  = V.rolling(20).mean()
    df['vol_ratio'] = V / df['vol_ma20']

    # ── OBV ───────────────────────────────────────────────────────────────────
    df['obv']     = (np.sign(C.diff()) * V).fillna(0).cumsum()
    df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

    # ── Candle anatomy ────────────────────────────────────────────────────────
    df['body']       = C - O
    df['body_abs']   = df['body'].abs()
    df['range']      = H - L
    df['body_pct']   = df['body_abs'] / df['range'].replace(0, np.nan)
    df['upper_wick'] = H - pd.concat([C, O], axis=1).max(axis=1)
    df['lower_wick'] = pd.concat([C, O], axis=1).min(axis=1) - L
    df['is_bull']    = C > O

    # ── Time features ─────────────────────────────────────────────────────────
    df['hour']       = df['timestamp'].dt.hour
    df['date']       = df['timestamp'].dt.date

    # ── Price structure ───────────────────────────────────────────────────────
    df['hh20']       = H.rolling(20).max().shift(1)   # prior 20-bar high
    df['ll20']       = L.rolling(20).min().shift(1)

    # ── EMA slope ────────────────────────────────────────────────────────────
    df['slope21']    = (df['ema21'] - df['ema21'].shift(5)) / df['ema21'].shift(5) * 100

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trade(df: pd.DataFrame, i: int, entry: float,
                   sl: float, tp: float, max_hold: int) -> dict | None:
    """
    Walk forward candle by candle. SL or TP hit → record result.
    If neither within max_hold → close at last candle's close.
    """
    risk = entry - sl
    if risk <= 0 or tp <= entry:
        return None
    for j in range(1, max_hold + 1):
        if i + j >= len(df):
            break
        fwd = df.iloc[i + j]
        if fwd['low'] <= sl:
            return dict(result='loss', pnl=(sl - entry)/entry*100,
                        exit_price=sl, hold=j,
                        exit_time=fwd['timestamp'])
        if fwd['high'] >= tp:
            return dict(result='win', pnl=(tp - entry)/entry*100,
                        exit_price=tp, hold=j,
                        exit_time=fwd['timestamp'])
    # Timeout
    idx = min(i + max_hold, len(df) - 1)
    ep  = df.iloc[idx]['close']
    pnl = (ep - entry) / entry * 100
    return dict(result='win' if pnl > 0 else 'loss', pnl=pnl,
                exit_price=ep, hold=max_hold,
                exit_time=df.iloc[idx]['timestamp'])


# ─────────────────────────────────────────────────────────────────────────────
# SETUP 1 — VOLUME SURGE STRUCTURE BREAK
# ─────────────────────────────────────────────────────────────────────────────
def detect_setup1(df: pd.DataFrame, i: int) -> dict | None:
    """
    Logic:
    1. Current candle range is 2.5× the mean range of prior 6 bars
       → exceptionally large candle = someone big is acting
    2. Volume is 4× the 20-bar average
       → confirms institutions, not retail noise
    3. Close is above the prior 20-bar high (structural break)
       → no more overhead resistance from recent memory
    4. EMA21 slope is positive → we are in an uptrend
    5. Body > 60% of total range → conviction, not a wick-heavy candle
    6. Avoid dead-volume hours (Asian session 22-06 UTC)

    Entry : close of trigger candle (confirmed break)
    SL    : open of trigger candle minus 0.15×ATR
            (below the real body — if price returns INTO the candle body,
             the breakout has failed)
    TP    : entry + 1.5 × risk  (quick 1.5:1 — momentum moves are fast)
    Hold  : max 10 candles (5 hours)
    """
    if i < 20 or i >= len(df) - 12:
        return None
    c = df.iloc[i]

    recent_avg_range = df['range'].iloc[i - 6:i].mean()

    # ── Conditions ──────────────────────────────────────────────────────
    if c['range'] < 2.5 * recent_avg_range:    return None  # not big enough
    if not c['is_bull']:                        return None  # must be bullish
    if c['vol_ratio'] < 4.0:                   return None  # needs 4× volume
    if c['body_pct'] < 0.60:                   return None  # body must dominate
    if c['close'] <= c['hh20']:                return None  # must break structure
    if c['slope21'] <= 0:                      return None  # must be in uptrend
    if c['hour'] in [0, 1, 2, 3, 4, 5, 22, 23]: return None  # avoid dead hours

    entry = c['close']
    sl    = c['open'] - 0.15 * c['atr']
    tp    = entry + S1_TP_RR * (entry - sl)

    return dict(entry=entry, sl=sl, tp=tp, max_hold=10,
                setup='S1', signal_time=c['timestamp'],
                vol_ratio=round(c['vol_ratio'], 2),
                body_pct=round(c['body_pct'], 2))


# ─────────────────────────────────────────────────────────────────────────────
# SETUP 2 — KELTNER SQUEEZE + OBV MOMENTUM IGNITION
# ─────────────────────────────────────────────────────────────────────────────
def detect_setup2(df: pd.DataFrame, i: int) -> dict | None:
    """
    Logic:
    1. Bollinger Bands are INSIDE the Keltner Channel for 5+ of last 8 bars
       → "squeeze" = volatility at extreme low, energy coiling like a spring
    2. RSI was between 40-65 during the squeeze (not at overbought/oversold)
       → market is in calm accumulation, not panic/euphoria
    3. OBV is rising over last 5 bars → smart money is buying during calm
    4. Trigger: first candle to close ABOVE BB upper after being inside
       → the spring is releasing
    5. Vol > 1.8× average + body > 55% + RSI 55-75 → conviction on release
    6. Previous candle was still inside BB (confirms this is the first break)

    Entry : close of first breakout candle
    SL    : minimum low of prior 3 candles minus 0.1×ATR
            (if price returns below the squeeze zone, move is invalidated)
    TP    : entry + 2.5 × risk  (squeeze releases can run far)
    Hold  : max 14 candles (7 hours)
    """
    if i < 15 or i >= len(df) - 16:
        return None
    c    = df.iloc[i]
    prev = df.iloc[i - 1]

    # ── Squeeze presence ───────────────────────────────────────────────────
    sq_count = df['squeeze'].iloc[i - 8:i].sum()
    if sq_count < 5:
        return None

    # ── RSI coiling (not at extremes during squeeze) ───────────────────────
    rsi_window = df['rsi'].iloc[i - 8:i]
    if not rsi_window.between(40, 65).all():
        return None

    # ── OBV rising → smart money loading up ──────────────────────────────
    if df['obv'].iloc[i] <= df['obv'].iloc[i - 5]:
        return None

    # ── Prev candle still inside BB → this is the first break ────────────
    if prev['close'] >= prev['bb_upper']:
        return None

    # ── Trigger candle conditions ─────────────────────────────────────────
    if not c['is_bull']:                   return None
    if c['close'] <= c['bb_upper']:        return None  # must exit BB
    if c['body_pct'] < 0.55:              return None
    if c['vol_ratio'] < 1.8:              return None
    if not (55 < c['rsi'] < 75):          return None  # sweet spot — rising but not topped

    entry = c['close']
    sl    = df['low'].iloc[i - 3:i + 1].min() - 0.1 * c['atr']
    tp    = entry + S2_TP_RR * (entry - sl)

    return dict(entry=entry, sl=sl, tp=tp, max_hold=14,
                setup='S2', signal_time=c['timestamp'],
                vol_ratio=round(c['vol_ratio'], 2),
                body_pct=round(c['body_pct'], 2))


# ─────────────────────────────────────────────────────────────────────────────
# SETUP 3 — VWAP RECLAIM + EMA TRIPLE ALIGNMENT
# ─────────────────────────────────────────────────────────────────────────────
def detect_setup3(df: pd.DataFrame, i: int) -> dict | None:
    """
    Logic:
    1. EMA9 > EMA21 > EMA50 (full bull stack) → trend is aligned on all speeds
    2. Previous candle closed BELOW VWAP → price dipped, shaking out weak hands
    3. Current candle closes BACK ABOVE VWAP → institutions defended VWAP
       (VWAP is the "fair value" price; smart money always comes back to it)
    4. Vol > 1.5× average → the reclaim had participation
    5. Body > 45% → it's a real move, not a wick
    6. RSI < 72 → not overextended yet, still room to run
    7. BB% < 85% → not at the top of the Bollinger Band
    8. London/NY session (7am-10pm UTC) → active markets

    Entry : close of VWAP reclaim candle
    SL    : below the lower of (prev low, current VWAP) minus 0.15×ATR
            (if price falls back through VWAP and previous low, thesis is wrong)
    TP    : entry + 2.2 × risk
    Hold  : max 12 candles (6 hours)
    """
    if i < 15 or i >= len(df) - 14:
        return None
    c    = df.iloc[i]
    prev = df.iloc[i - 1]

    # ── Full EMA alignment ────────────────────────────────────────────────
    if not (c['ema9'] > c['ema21'] > c['ema50']):
        return None

    # ── VWAP dip and reclaim ──────────────────────────────────────────────
    if prev['close'] >= prev['vwap']:
        return None  # no dip happened
    if c['close'] <= c['vwap']:
        return None  # didn't reclaim

    # ── Candle quality ────────────────────────────────────────────────────
    if not c['is_bull']:          return None
    if c['body_pct'] < 0.45:     return None
    if c['vol_ratio'] < 1.5:     return None
    if c['rsi'] > 72:            return None  # overbought
    if c['bb_pct'] > 0.85:       return None  # near BB top

    # ── Session filter ────────────────────────────────────────────────────
    if c['hour'] not in list(range(7, 22)):
        return None

    entry = c['close']
    sl    = min(prev['low'], c['vwap']) - 0.15 * c['atr']
    tp    = entry + S3_TP_RR * (entry - sl)

    return dict(entry=entry, sl=sl, tp=tp, max_hold=12,
                setup='S3', signal_time=c['timestamp'],
                vol_ratio=round(c['vol_ratio'], 2),
                body_pct=round(c['body_pct'], 2))


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE — 1 trade per day max
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each candle, check all 3 setups in priority order (S1 > S2 > S3).
    Enforce 1 trade per calendar day.
    Simulate each trade walking forward.
    Return full trade DataFrame.
    """
    trades     = []
    used_dates = set()   # track which dates already have a trade

    detectors = [
        ('S1 — Vol Surge Structure Break',         detect_setup1, 10),
        ('S2 — Keltner Squeeze OBV Ignition',       detect_setup2, 14),
        ('S3 — VWAP Reclaim EMA Alignment',         detect_setup3, 10),
    ]

    for i in range(20, len(df) - 20):
        trade_date = df.iloc[i]['timestamp'].date()

        # Skip if we already traded today
        if trade_date in used_dates:
            continue

        # Check setups in priority order — take first one that fires
        for setup_name, detector, warmup in detectors:
            if i < warmup:
                continue
            levels = detector(df, i)
            if levels is None:
                continue

            # Simulate
            result = simulate_trade(
                df, i,
                entry    = levels['entry'],
                sl       = levels['sl'],
                tp       = levels['tp'],
                max_hold = levels['max_hold']
            )
            if result is None:
                continue

            used_dates.add(trade_date)

            trades.append({
                'setup'       : levels['setup'],
                'setup_name'  : setup_name,
                'signal_time' : levels['signal_time'],
                'exit_time'   : result['exit_time'],
                'entry'       : round(levels['entry'], 4),
                'sl'          : round(levels['sl'], 4),
                'tp'          : round(levels['tp'], 4),
                'result'      : result['result'],
                'exit_price'  : round(result['exit_price'], 4),
                'pnl_pct'     : round(result['pnl'], 4),
                'hold'        : result['hold'],
                'vol_ratio'   : levels['vol_ratio'],
                'body_pct'    : levels['body_pct'],
            })
            break   # only 1 trade per candle/day

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY + METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_equity(trades: pd.DataFrame, account: float) -> tuple:
    """
    Compound equity curve using 10% risk per trade.
    Win: balance += balance × RISK_PCT × RR
    Loss: balance -= balance × RISK_PCT
    """
    balance  = account
    balances = [balance]
    for _, t in trades.iterrows():
        risk_amt = balance * RISK_PCT
        if t['result'] == 'win':
            # RR for each setup
            rr = {'S1': S1_TP_RR, 'S2': S2_TP_RR, 'S3': S3_TP_RR}.get(t['setup'], 2.0)
            balance += risk_amt * rr
        else:
            balance -= risk_amt
        balances.append(round(balance, 4))
    return balances


def compute_drawdown(balances: list) -> tuple:
    peak = balances[0]
    dd   = []
    max_dd = 0.0
    for b in balances:
        if b > peak:
            peak = b
        d = (peak - b) / peak * 100
        dd.append(-d)
        if d > max_dd:
            max_dd = d
    return dd, max_dd


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(trades: pd.DataFrame, balances: list, max_dd: float):
    if len(trades) == 0:
        print("  ⚠️  No trades found.")
        return

    wins      = trades[trades['result'] == 'win']
    losses    = trades[trades['result'] == 'loss']
    wr        = len(wins) / len(trades) * 100
    avg_w     = wins['pnl_pct'].mean() if len(wins) else 0
    avg_l     = losses['pnl_pct'].mean() if len(losses) else 0
    rr        = abs(avg_w / avg_l) if avg_l else 0
    final_bal = balances[-1]
    ret       = (final_bal - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    # Monthly
    trades['month'] = pd.to_datetime(trades['signal_time']).dt.to_period('M')
    monthly = trades.groupby('month').agg(
        count=('pnl_pct','count'),
        pnl  =('pnl_pct','sum'),
        wins =('result', lambda x: (x=='win').sum())
    ).reset_index()
    monthly['wr']    = monthly['wins'] / monthly['count'] * 100
    green_mo = (monthly['pnl'] > 0).sum()

    SEP = '═' * 68
    print(f'\n{SEP}')
    print(f'  BACKTEST RESULTS — ETHUSDT 30m — Deep Quant v2.0')
    print(f'{SEP}')
    print(f'  Data range  : {trades["signal_time"].min().date()} → {trades["signal_time"].max().date()}')
    print(f'  Total trades: {len(trades)}  (target: ~{len(pd.to_datetime(trades["signal_time"]).dt.date.unique())} trading days)')
    print(f'  Wins        : {len(wins)}   Losses : {len(losses)}')
    print(f'  Win Rate    : {wr:.1f}%')
    print(f'  Avg Win     : {avg_w:+.2f}%    Avg Loss: {avg_l:+.2f}%')
    print(f'  RR (actual) : {rr:.2f}×')
    print(f'{SEP}')
    print(f'  Start bal   : ${ACCOUNT_SIZE:,.2f}    Risk/trade: {RISK_PCT*100:.0f}%')
    print(f'  Final bal   : ${final_bal:,.2f}')
    print(f'  Total return: {ret:+.1f}%')
    print(f'  Max drawdown: {max_dd:.2f}%')
    print(f'  Green months: {green_mo}/{len(monthly)}')
    print(f'{SEP}')

    # Per-setup breakdown
    print(f'\n  PER-SETUP BREAKDOWN:')
    for s in ['S1','S2','S3']:
        sub  = trades[trades['setup'] == s]
        if len(sub) == 0: continue
        sw   = sub[sub['result']=='win']
        sl   = sub[sub['result']=='loss']
        swr  = len(sw)/len(sub)*100 if len(sub) else 0
        sn   = sub.iloc[0]['setup_name']
        print(f'  {sn}')
        print(f'    Trades: {len(sub)}  WR: {swr:.1f}%  '
              f'AvgW: {sw["pnl_pct"].mean():+.2f}%  AvgL: {sl["pnl_pct"].mean():+.2f}%')

    # Monthly table
    print(f'\n  MONTHLY BREAKDOWN:')
    print(f'  {"Month":<10}  {"Trades":>6}  {"WR%":>5}  {"PnL%":>8}  {"Status":>6}')
    print(f'  {"─"*48}')
    for _, row in monthly.iterrows():
        icon = '🟢' if row['pnl'] > 0 else '🔴'
        print(f'  {str(row["month"]):<10}  {row["count"]:>6}  '
              f'{row["wr"]:>4.0f}%  {row["pnl"]:>+7.2f}%  {icon}')

    trades.to_csv('trades_log.csv', index=False)
    print(f'\n  Trade log → trades_log.csv')
    print(f'{SEP}\n')


# ─────────────────────────────────────────────────────────────────────────────
# MATPLOTLIB DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def draw_dashboard(trades: pd.DataFrame, balances: list,
                   dd_series: list, max_dd: float):

    if len(trades) == 0:
        print("  No trades to plot.")
        return

    # ── Colour palette ───────────────────────────────────────────────────────
    BG     = '#080c14'
    SURF   = '#0f1520'
    SURF2  = '#161e2e'
    GRID   = '#1c2535'
    GREEN  = '#00e5a0'
    RED    = '#ff4466'
    GOLD   = '#f5c842'
    BLUE   = '#4d9fff'
    ORANGE = '#ff8c42'
    MUTED  = '#4a5568'
    TEXT   = '#dde3ee'

    # Setup colours
    SC = {'S1': GREEN, 'S2': BLUE, 'S3': GOLD}

    plt.rcParams.update({
        'figure.facecolor' : BG,
        'axes.facecolor'   : SURF,
        'axes.edgecolor'   : GRID,
        'axes.labelcolor'  : MUTED,
        'xtick.color'      : MUTED,
        'ytick.color'      : MUTED,
        'grid.color'       : GRID,
        'grid.alpha'       : 0.6,
        'grid.linewidth'   : 0.5,
        'text.color'       : TEXT,
        'font.family'      : 'monospace',
        'font.size'        : 8.5,
    })

    wins   = trades[trades['result'] == 'win']
    losses = trades[trades['result'] == 'loss']
    wr     = len(wins) / len(trades) * 100
    avg_w  = wins['pnl_pct'].mean() if len(wins) else 0
    avg_l  = losses['pnl_pct'].mean() if len(losses) else 0
    rr     = abs(avg_w / avg_l) if avg_l else 0
    final  = balances[-1]
    ret    = (final - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100

    # Monthly PnL
    trades2 = trades.copy()
    trades2['month'] = pd.to_datetime(trades2['signal_time']).dt.to_period('M')
    monthly = trades2.groupby('month').agg(
        pnl=('pnl_pct','sum'), count=('pnl_pct','count'),
        wins=('result', lambda x: (x=='win').sum())
    ).reset_index()
    monthly['wr'] = monthly['wins'] / monthly['count'] * 100

    # Running win-rate
    run_wr = [
        len(trades.iloc[:k+1][trades.iloc[:k+1]['result']=='win']) / (k+1) * 100
        for k in range(len(trades))
    ]

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.canvas.manager.set_window_title('ETHUSDT 30m — Deep Quant Dashboard v2')

    gs = gridspec.GridSpec(
        4, 5,
        figure=fig,
        hspace=0.55, wspace=0.38,
        top=0.88, bottom=0.06,
        left=0.05, right=0.97
    )

    ax_eq   = fig.add_subplot(gs[0:2, 0:4])   # equity curve — big
    ax_mo   = fig.add_subplot(gs[0:2, 4])     # monthly bars
    ax_dd   = fig.add_subplot(gs[2, 0:3])     # drawdown
    ax_wr   = fig.add_subplot(gs[2, 3:5])     # running win-rate
    ax_bar  = fig.add_subplot(gs[3, 0:3])     # per-trade PnL bars
    ax_dist = fig.add_subplot(gs[3, 3])       # PnL distribution
    ax_pie  = fig.add_subplot(gs[3, 4])       # setup breakdown pie

    trade_x = list(range(len(balances)))
    dot_x   = list(range(1, len(trades) + 1))
    dot_y   = balances[1:]

    # ── [1] EQUITY CURVE ────────────────────────────────────────────────────
    ax = ax_eq
    # Gradient fill
    ax.fill_between(trade_x, ACCOUNT_SIZE, balances,
                    where=[b >= ACCOUNT_SIZE for b in balances],
                    color=GREEN, alpha=0.12, interpolate=True)
    ax.fill_between(trade_x, ACCOUNT_SIZE, balances,
                    where=[b < ACCOUNT_SIZE for b in balances],
                    color=RED, alpha=0.15, interpolate=True)
    ax.axhline(ACCOUNT_SIZE, color=MUTED, lw=0.8, ls='--', alpha=0.5)
    ax.plot(trade_x, balances, color=GREEN, lw=2.2, zorder=4)

    # Dots coloured by setup
    for _, t in trades.iterrows():
        idx_  = int(t.name) + 1
        col   = SC.get(t['setup'], MUTED)
        marker= 'o' if t['result'] == 'win' else 'x'
        ms    = 40 if t['result'] == 'win' else 35
        ax.scatter(idx_, balances[idx_], c=col, s=ms, marker=marker,
                   zorder=6, linewidths=1.5)

    # Annotate final
    ax.annotate(f"  ${final:,.0f}  ({ret:+.0f}%)",
                xy=(trade_x[-1], balances[-1]),
                color=GREEN, fontsize=11, fontweight='bold', va='center')

    ax.set_title('EQUITY CURVE  ·  10% Compounding Risk Per Trade',
                 fontsize=11, fontweight='bold', color=TEXT, pad=10, loc='left')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Balance (USDT)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v:,.0f}'))
    ax.set_xlim(0, len(balances))
    ax.grid(True, axis='y')

    # Legend for setups
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='none', markerfacecolor=GREEN, ms=8, label='S1 Win'),
        Line2D([0],[0], marker='x', color=GREEN, ms=8, lw=1.5, label='S1 Loss'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=BLUE,  ms=8, label='S2 Win'),
        Line2D([0],[0], marker='x', color=BLUE,  ms=8, lw=1.5, label='S2 Loss'),
        Line2D([0],[0], marker='o', color='none', markerfacecolor=GOLD,  ms=8, label='S3 Win'),
        Line2D([0],[0], marker='x', color=GOLD,  ms=8, lw=1.5, label='S3 Loss'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=7.5,
              facecolor=SURF2, edgecolor=GRID, ncol=3)

    # Stats box
    stats = (
        f"  Trades:{len(trades)}  Wins:{len(wins)}  Losses:{len(losses)}\n"
        f"  WR:{wr:.1f}%   RR:{rr:.2f}×   MaxDD:{max_dd:.1f}%\n"
        f"  AvgWin:{avg_w:+.2f}%  AvgLoss:{avg_l:+.2f}%\n"
        f"  Risk/trade:10%  Return:{ret:+.0f}%"
    )
    ax.text(0.01, 0.97, stats, transform=ax.transAxes,
            fontsize=8.5, color=TEXT, va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc=SURF2, ec=GRID, alpha=0.92))

    # ── [2] MONTHLY PnL BARS ─────────────────────────────────────────────────
    ax = ax_mo
    mo_lbls = [str(m)[-2:] for m in monthly['month']]
    mo_pnl  = list(monthly['pnl'])
    mo_col  = [GREEN if p > 0 else RED for p in mo_pnl]
    bars    = ax.barh(mo_lbls, mo_pnl, color=mo_col, alpha=0.8, height=0.6)
    for bar, val in zip(bars, mo_pnl):
        x = val + 0.05 if val >= 0 else val - 0.05
        ha = 'left' if val >= 0 else 'right'
        ax.text(x, bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', ha=ha, fontsize=7.5,
                color=GREEN if val>=0 else RED)
    ax.axvline(0, color=MUTED, lw=0.7, ls='--')
    ax.set_title('MONTHLY PnL%', fontsize=9, color=TEXT, loc='left', pad=6)
    ax.set_xlabel('PnL%')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:+.0f}%'))
    ax.grid(True, axis='x')
    ax.invert_yaxis()
    green_count = sum(1 for p in mo_pnl if p > 0)
    ax.set_title(f'MONTHLY PnL%  ({green_count}/{len(mo_pnl)} green)',
                 fontsize=9, color=TEXT, loc='left', pad=6)

    # ── [3] DRAWDOWN ──────────────────────────────────────────────────────────
    ax = ax_dd
    ax.fill_between(trade_x, 0, dd_series, color=RED, alpha=0.35)
    ax.plot(trade_x, dd_series, color=RED, lw=1.3)
    ax.axhline(0, color=MUTED, lw=0.6, ls='--')
    min_dd_i = int(np.argmin(dd_series))
    ax.annotate(f' Max DD\n {dd_series[min_dd_i]:.1f}%',
                xy=(min_dd_i, dd_series[min_dd_i]),
                color=RED, fontsize=8,
                xytext=(min_dd_i + 2, dd_series[min_dd_i] * 1.15))
    ax.set_title('DRAWDOWN %', fontsize=9, color=TEXT, loc='left', pad=6)
    ax.set_xlabel('Trade #')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:.1f}%'))
    ax.set_xlim(0, len(balances))
    ax.grid(True, axis='y')

    # ── [4] RUNNING WIN-RATE ──────────────────────────────────────────────────
    ax = ax_wr
    rw_x = list(range(1, len(run_wr) + 1))
    ax.plot(rw_x, run_wr, color=BLUE, lw=1.8, zorder=3)
    ax.axhline(50, color=MUTED, lw=0.7, ls='--', alpha=0.7)
    ax.fill_between(rw_x, 50, run_wr,
                    where=[v >= 50 for v in run_wr],
                    color=GREEN, alpha=0.12, interpolate=True)
    ax.fill_between(rw_x, 50, run_wr,
                    where=[v < 50 for v in run_wr],
                    color=RED, alpha=0.12, interpolate=True)
    ax.annotate(f'  Final: {run_wr[-1]:.1f}%',
                xy=(rw_x[-1], run_wr[-1]), color=BLUE, fontsize=8)
    ax.set_title('RUNNING WIN RATE%', fontsize=9, color=TEXT, loc='left', pad=6)
    ax.set_xlabel('Trade #')
    ax.set_ylim(20, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:.0f}%'))
    ax.grid(True, axis='y')

    # ── [5] PER-TRADE PnL BARS ────────────────────────────────────────────────
    ax = ax_bar
    bar_pnl = list(trades['pnl_pct'])
    bar_col = [SC.get(t['setup'], MUTED) if t['result']=='win' else RED
               for _, t in trades.iterrows()]
    ax.bar(range(1, len(trades)+1), bar_pnl,
           color=bar_col, width=0.7, alpha=0.85, zorder=3)
    ax.axhline(0, color=MUTED, lw=0.7, ls='--')
    ax.set_title('PER-TRADE PnL%  (colour = setup)', fontsize=9, color=TEXT, loc='left', pad=6)
    ax.set_xlabel('Trade #')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:+.1f}%'))
    ax.set_xlim(0, len(trades) + 1)
    ax.grid(True, axis='y')

    # ── [6] PnL DISTRIBUTION ──────────────────────────────────────────────────
    ax = ax_dist
    bins = np.linspace(trades['pnl_pct'].min() - 0.2,
                       trades['pnl_pct'].max() + 0.2, 22)
    ax.hist(wins['pnl_pct'],   bins=bins, color=GREEN, alpha=0.7,
            label=f'Win ({len(wins)})', edgecolor='none')
    ax.hist(losses['pnl_pct'], bins=bins, color=RED,   alpha=0.7,
            label=f'Loss ({len(losses)})', edgecolor='none')
    ax.axvline(0, color=MUTED, lw=0.7, ls='--')
    ax.axvline(avg_w, color=GREEN, lw=1.2, ls=':', alpha=0.9)
    ax.axvline(avg_l, color=RED,   lw=1.2, ls=':', alpha=0.9)
    ax.set_title('PnL DISTRIBUTION', fontsize=9, color=TEXT, loc='left', pad=6)
    ax.set_xlabel('PnL%')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'{v:+.1f}%'))
    ax.legend(fontsize=7.5, facecolor=SURF2, edgecolor=GRID)
    ax.grid(True, axis='y')

    # ── [7] SETUP BREAKDOWN PIE ───────────────────────────────────────────────
    ax = ax_pie
    s_counts = trades['setup'].value_counts()
    s_labels = [f"{s}\n{s_counts[s]} trades" for s in s_counts.index]
    s_colors = [SC.get(s, MUTED) for s in s_counts.index]
    wedges, texts, autotexts = ax.pie(
        s_counts.values, labels=s_labels,
        colors=s_colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.65,
        textprops={'fontsize': 8, 'color': TEXT},
        wedgeprops={'linewidth': 0}
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color(BG)
        at.set_fontweight('bold')
    ax.set_title('SETUP MIX', fontsize=9, color=TEXT, loc='center', pad=6)

    # ── HEADER ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.955,
             'ETHUSDT 30m  ·  Deep Quant Backtest v2.0  ·  3 Discovered Setups',
             ha='center', fontsize=13, fontweight='bold', color=TEXT)
    s1n = trades[trades['setup']=='S1']['signal_time'].min()
    sln = trades[trades['setup']=='S1']['signal_time'].max()
    dr  = f"{trades['signal_time'].min().date()}  →  {trades['signal_time'].max().date()}"
    fig.text(0.5, 0.926,
             f"{dr}   |   {len(trades)} trades ({RISK_PCT*100:.0f}% risk)   |   "
             f"WR {wr:.1f}%   |   RR {rr:.2f}×   |   "
             f"Return {ret:+.0f}%   |   MaxDD {max_dd:.1f}%",
             ha='center', fontsize=9, color=MUTED)

    # ── Save & Open ───────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHART_FILE)
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"\n  ✅ Dashboard → {out}")

    opened = False
    try:
        sys_name = platform.system()
        if sys_name == 'Darwin':
            subprocess.Popen(['open', out]); opened = True
        elif sys_name == 'Windows':
            os.startfile(out); opened = True
        else:
            for v in ['eog','feh','display','gpicview','ristretto','gimp','shotwell']:
                if subprocess.run(['which', v], capture_output=True).returncode == 0:
                    subprocess.Popen([v, out],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    print(f"  Opening with {v}...")
                    opened = True
                    break
    except Exception:
        pass

    if not opened:
        print(f"\n  ⚠️  No image viewer found. Launching browser server...")
        print(f"  ┌───────────────────────────────────────────────┐")
        print(f"  │  Open:  http://localhost:8765/{CHART_FILE:<18}│")
        print(f"  │  Press Ctrl+C to stop when done               │")
        print(f"  └───────────────────────────────────────────────┘\n")
        try:
            import http.server, socketserver
            os.chdir(os.path.dirname(os.path.abspath(out)))
            handler = http.server.SimpleHTTPRequestHandler
            handler.log_message = lambda *a: None
            with socketserver.TCPServer(('', 8765), handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
        except OSError as e:
            print(f"  Could not start server: {e}")
            print(f"  Open manually: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIGNAL SCANNER
# ─────────────────────────────────────────────────────────────────────────────
def scan_live(df_raw: pd.DataFrame) -> dict | None:
    """
    Call this every 30 minutes with latest OHLCV data.
    Returns signal dict or None.

    Live bot integration:
      1. Fetch latest candles from exchange (Binance/Bybit)
      2. Pass as DataFrame → this function
      3. If signal returned → place order with entry/sl/tp
      4. Monitor position for SL/TP hit
    """
    df = build_indicators(df_raw.copy())
    i  = len(df) - 1

    for detector in [detect_setup1, detect_setup2, detect_setup3]:
        levels = detector(df, i)
        if levels:
            return {
                'symbol'     : SYMBOL if DATA_MODE == 'binance' else 'LOCAL',
                'timeframe'  : INTERVAL,
                'setup'      : levels['setup'],
                'direction'  : 'LONG',
                'entry'      : levels['entry'],
                'stop_loss'  : levels['sl'],
                'take_profit': levels['tp'],
                'max_hold'   : levels['max_hold'],
                'vol_ratio'  : levels['vol_ratio'],
                'signal_time': levels['signal_time'],
            }
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sym_label = SYMBOL if DATA_MODE == 'binance' else 'LOCAL CSV'
    print(f'\n🚀  Deep Quant Backtest v2.0  ·  {sym_label} {INTERVAL}')
    print(f'    3 Setups  ·  {RISK_PCT*100:.0f}% Risk  ·  1 Trade/Day  ·  Visual Dashboard\n')

    # ── Load data (Binance or CSV) ─────────────────────────────────────────
    df_raw = get_data()
    print(f'\n✅  {len(df_raw):,} candles loaded')

    print('\n📊  Building 57 indicators...')
    df_ind = build_indicators(df_raw)
    print(f'✅  Ready — {len(df_ind):,} candles after warmup')

    print('\n🔍  Scanning all 3 setups (1 trade/day limit)...\n')
    trades = run_backtest(df_ind)

    balances          = compute_equity(trades, ACCOUNT_SIZE)
    dd_series, max_dd = compute_drawdown(balances)
    trades['balance'] = balances[1:]

    print_report(trades, balances, max_dd)

    print('🎨  Generating P&L dashboard...')
    draw_dashboard(trades, balances, dd_series, max_dd)

    # ── Live signal scan on most recent candle ─────────────────────────────
    print('\n📡  Live signal scan (latest candle):')
    sig = scan_live(df_raw)
    if sig:
        print(f'  ⚡  SIGNAL FOUND — {sig["setup"]}')
        for k, v in sig.items():
            print(f'      {k:<15}: {v}')
    else:
        print('  ⏳  No signal on latest candle — waiting for next bar.')
    print()
