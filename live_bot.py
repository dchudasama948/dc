import ccxt
import pandas as pd
import time
import joblib
import os
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

exchange = ccxt.binance({'enableRateLimit': True})
symbol = 'WIF/USDT'
timeframe = '1m'
limit = 1500
quantity = 100

# Load or create models
def load_or_initialize_models():
    model_path = 'models'
    try:
        clf = joblib.load(f"{model_path}/entry_model_ensemble.pkl")
        tp_model = joblib.load(f"{model_path}/tp_model.pkl")
        sl_model = joblib.load(f"{model_path}/sl_model.pkl")
        print("[MODEL] Loaded existing models.")
    except:
        clf = VotingClassifier(estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
            ('lgb', LGBMClassifier())
        ], voting='soft')
        tp_model = RandomForestRegressor()
        sl_model = RandomForestRegressor()
        print("[MODEL] Initialized new models.")
    return clf, tp_model, sl_model

clf, tp_model, sl_model = load_or_initialize_models()

def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def compute_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['atr'] = df['high'].combine(df['low'], max) - df['low'].combine(df['close'].shift(), min)
    df['atr'] = df['atr'].rolling(14).mean()
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    return df.dropna()

def label_data(df):
    df['future_close'] = df['close'].shift(-5)
    df['label'] = (df['future_close'] > df['close'] * 1.002).astype(int)
    return df.dropna()

def train_models(df):
    features = df[['ema_fast', 'ema_slow', 'rsi', 'atr', 'macd', 'signal', 'bb_mid', 'bb_upper', 'bb_lower']]
    X = features.values
    y = df['label'].values
    clf.fit(X, y)
    tp_model.fit(X, (df['future_close'] / df['close'] - 1).clip(0, 0.02).values)
    sl_model.fit(X, (1 - df['future_close'] / df['close']).clip(0, 0.02).values)
    joblib.dump(clf, "models/entry_model_ensemble.pkl")
    joblib.dump(tp_model, "models/tp_model.pkl")
    joblib.dump(sl_model, "models/sl_model.pkl")

def simulate_trade(row, log, current_position, open_trade):
    features = row[['ema_fast', 'ema_slow', 'rsi', 'atr', 'macd', 'signal', 'bb_mid', 'bb_upper', 'bb_lower']].values.reshape(1, -1)
    prob = clf.predict_proba(features)[0][1]
    if prob < 0.7:
        return current_position, open_trade

    direction = 'buy' if row['ema_fast'] > row['ema_slow'] else 'sell'
    entry = row['close']
    tp_pct = tp_model.predict(features)[0]
    sl_pct = sl_model.predict(features)[0]
    tp = entry * (1 + tp_pct) if direction == 'buy' else entry * (1 - tp_pct)
    sl = entry * (1 - sl_pct) if direction == 'buy' else entry * (1 + sl_pct)

    open_trade = {
        'time': row.name,
        'direction': direction,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'qty': quantity,
        'tsl_pct': 0.0015
    }
    current_position = direction
    print(f"[TRADE OPEN] {open_trade}")
    log.append(open_trade)
    return current_position, open_trade

def check_trade_exit(row, log, current_position, open_trade):
    if not open_trade:
        return current_position, open_trade

    price = row['close']
    entry = open_trade['entry']
    tp = open_trade['tp']
    sl = open_trade['sl']
    direction = open_trade['direction']
    tsl_pct = open_trade['tsl_pct']
    exit = None

    if direction == 'buy':
        if price >= tp:
            exit = 'TP'
        elif price <= sl:
            exit = 'SL'
        elif price > entry:
            new_sl = price * (1 - tsl_pct)
            if new_sl > sl:
                open_trade['sl'] = new_sl
    else:
        if price <= tp:
            exit = 'TP'
        elif price >= sl:
            exit = 'SL'
        elif price < entry:
            new_sl = price * (1 + tsl_pct)
            if new_sl < sl:
                open_trade['sl'] = new_sl

    if exit:
        pnl = (price - entry) * quantity if direction == 'buy' else (entry - price) * quantity
        closed = {**open_trade, 'exit_price': price, 'exit_time': row.name, 'pnl': pnl, 'exit': exit}
        print(f"[TRADE CLOSE] {closed}")
        log.append(closed)
        current_position = None
        open_trade = None
    return current_position, open_trade

# Bot main loop
log = []
current_position = None
open_trade = None
print("[BOT] Running Auto-Trained AI Scalping Bot")

try:
    while True:
        df = fetch_data()
        df = compute_indicators(df)
        df = label_data(df)
        train_models(df)
        row = df.iloc[-1]
        current_position, open_trade = check_trade_exit(row, log, current_position, open_trade)
        if current_position is None:
            current_position, open_trade = simulate_trade(row, log, current_position, open_trade)
        pd.DataFrame(log).to_csv("logs/live_trade_log.csv", index=False)
        time.sleep(5)

except KeyboardInterrupt:
    print("[BOT] Stopped by user")
    pd.DataFrame(log).to_csv("logs/live_trade_log.csv", index=False)