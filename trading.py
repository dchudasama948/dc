import ccxt
import pandas as pd
import time
import joblib
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Binance API configuration
BINANCE_API_KEY = 'YOUR_API_KEY_HERE'
BINANCE_API_SECRET = 'YOUR_API_SECRET_HERE'
LEVERAGE = 50  # Use 50x leverage
MARGIN_TYPE = 'CROSSED'  # Use CROSSED margin mode

# Trading parameters
symbol = 'WIF/USDT'
binance_symbol = 'WIFUSDT'  # Symbol format for Binance Futures API
timeframe = '1m'
limit = 500
quantity = 0.1  # Trade size in BTC (adjust based on your risk profile)

# Initialize Binance client
binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

# Initialize ccxt exchange for data fetching
exchange = ccxt.binance({
    'enableRateLimit': True
})

# Initialize or load models
try:
    clf = joblib.load("entry_model.pkl")
    tp_model = joblib.load("tp_model.pkl")
    sl_model = joblib.load("sl_model.pkl")
    print("[MODEL] Models loaded successfully.")
except:
    clf = RandomForestClassifier(n_estimators=100)
    tp_model = RandomForestRegressor()
    sl_model = RandomForestRegressor()
    print("[MODEL] Models not found. New instances created.")

# Logging storage
log = []

# Current position tracking
current_position = None
open_trade = None  # Track open trade data

# Set leverage and margin type for the symbol
def set_leverage_and_margin_type():
    try:
        print(f"Setting leverage to {LEVERAGE}x for {binance_symbol}")
        leverage_response = binance_client.futures_change_leverage(
            symbol=binance_symbol,
            leverage=LEVERAGE
        )
        
        print(f"Setting margin type to {MARGIN_TYPE} for {binance_symbol}")
        try:
            margin_type_response = binance_client.futures_change_margin_type(
                symbol=binance_symbol,
                marginType=MARGIN_TYPE
            )
        except BinanceAPIException as e:
            # Error code -4046 means already in the specified margin type
            if e.code == -4046:
                print(f"Margin type already set to {MARGIN_TYPE}")
            else:
                raise e
                
        print("Leverage and margin type set successfully")
    except Exception as e:
        print(f"Error setting leverage and margin type: {e}")

# Get quantity precision for the symbol
def get_quantity_precision(symbol):
    try:
        exchange_info = binance_client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter in symbol_info['filters']:
                    if filter['filterType'] == 'LOT_SIZE':
                        step_size = float(filter['stepSize'])
                        precision = 0
                        while step_size < 1:
                            step_size *= 10
                            precision += 1
                        return precision
    except Exception as e:
        print(f"Error getting quantity precision: {e}")
        return 3  # Default precision
    return 3  # Default precision

# Get price precision for the symbol
def get_price_precision(symbol):
    try:
        exchange_info = binance_client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                for filter in symbol_info['filters']:
                    if filter['filterType'] == 'PRICE_FILTER':
                        tick_size = float(filter['tickSize'])
                        precision = 0
                        while tick_size < 1:
                            tick_size *= 10
                            precision += 1
                        return precision
    except Exception as e:
        print(f"Error getting price precision: {e}")
        return 2  # Default precision
    return 2  # Default precision

# Check current position on Binance Futures
def get_current_position():
    try:
        positions = binance_client.futures_position_information(symbol=binance_symbol)
        for position in positions:
            if position['symbol'] == binance_symbol:
                position_amt = float(position['positionAmt'])
                if position_amt > 0:
                    return 'buy'
                elif position_amt < 0:
                    return 'sell'
        return None
    except Exception as e:
        print(f"Error getting position info: {e}")
        return None

# Get current price from market
def get_current_price():
    try:
        ticker = binance_client.futures_symbol_ticker(symbol=binance_symbol)
        return float(ticker['price'])
    except Exception as e:
        print(f"Error getting current price: {e}")
        # Fallback to recent data
        df = fetch_data()
        return df['close'].iloc[-1]

# Fetch historical data using ccxt
def fetch_data():
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# Calculate indicators
def compute_indicators(df):
    df['ema_fast'] = df['close'].ewm(span=10).mean()
    df['ema_slow'] = df['close'].ewm(span=21).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

# Auto-labeling function for training
def label_data(df):
    df['future_close'] = df['close'].shift(-5)
    df['label'] = (df['future_close'] > df['close'] * 1.002).astype(int)
    return df.dropna()

# Train models on-the-fly
def train_models(df):
    features = df[['ema_fast', 'ema_slow', 'rsi', 'atr']]
    X = features.values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    print("[AI] Classifier Accuracy:", accuracy_score(y_test, clf.predict(X_test)))
    
    # Train TP model - predicts percentage move upward
    tp_model.fit(X_train, (df['future_close'] / df['close'] - 1).clip(0, 0.01).values[:len(X_train)])
    
    # Train SL model - predicts percentage move downward
    sl_model.fit(X_train, (1 - df['future_close'] / df['close']).clip(0, 0.01).values[:len(X_train)])
    
    # Save models for persistence
    joblib.dump(clf, "entry_model.pkl")
    joblib.dump(tp_model, "tp_model.pkl")
    joblib.dump(sl_model, "sl_model.pkl")

# Execute a trade on Binance Futures using limit order with immediate execution
def execute_trade(direction, price, quantity):
    try:
        # Get price precision and quantity precision
        price_precision = get_price_precision(binance_symbol)
        quantity_precision = get_quantity_precision(binance_symbol)
        
        # Round quantity to proper precision
        rounded_quantity = round(quantity, quantity_precision)
        
        # Get current market price
        current_price = get_current_price()
        
        # Set limit price slightly better than market to ensure fast execution
        # For BUY: set limit slightly higher, for SELL: set limit slightly lower
        price_adjustment = 0.001  # 0.1% adjustment
        
        if direction == 'buy':
            limit_price = current_price * (1 + price_adjustment)
        else:  # sell
            limit_price = current_price * (1 - price_adjustment)
            
        # Round limit price to proper precision
        limit_price = round(limit_price, price_precision)
        
        # Get wallet balance before trade
        balance_before = float(binance_client.futures_account_balance()[0]['balance'])
        print(f"Balance before trade: {balance_before}")
        
        # Create limit order
        side = 'BUY' if direction == 'buy' else 'SELL'
        order_response = binance_client.futures_create_order(
            symbol=binance_symbol,
            type='LIMIT',
            side=side,
            quantity=rounded_quantity,
            price=limit_price,
            timeInForce='GTC'  # Good Till Cancelled
        )
        
        print(f"Limit order placed: {order_response}")
        
        # Wait for order to fill
        order_id = order_response['orderId']
        filled = False
        retries = 10  # More retries for limit orders
        
        while not filled and retries > 0:
            order_info = binance_client.futures_get_order(
                symbol=binance_symbol, 
                orderId=order_id
            )
            
            if order_info['status'] == 'FILLED':
                filled = True
                print("Limit order filled!")
            else:
                print(f"Waiting for limit order to fill. Status: {order_info['status']}")
                time.sleep(1)
                retries -= 1
        
        # If not filled after retries, cancel the order
        if not filled:
            print(f"Order not filled after {retries} retries. Cancelling...")
            binance_client.futures_cancel_order(
                symbol=binance_symbol,
                orderId=order_id
            )
            return None
            
        # Get order execution details
        executed_price = float(order_info['avgPrice'])
        executed_qty = float(order_info['executedQty'])
        
        # Calculate current balance
        balance_after = float(binance_client.futures_account_balance()[0]['balance'])
        realized_pnl = balance_after - balance_before
        
        return {
            'time': datetime.now(),
            'direction': direction,
            'entry': executed_price,
            'qty': executed_qty,
            'realized_pnl': realized_pnl,
            'balance_before': balance_before,
            'balance_after': balance_after
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to execute trade: {e}")
        return None

# Close a position using limit order
def close_position(current_direction):
    try:
        # Get position info
        positions = binance_client.futures_position_information(symbol=binance_symbol)
        position_amt = 0
        
        for position in positions:
            if position['symbol'] == binance_symbol:
                position_amt = float(position['positionAmt'])
                break
                
        if position_amt == 0:
            print("No position to close")
            return None
            
        # Get current market price
        current_price = get_current_price()
        
        # Determine closing side and price
        close_side = 'SELL' if position_amt > 0 else 'BUY'
        price_adjustment = 0.001  # 0.1% adjustment
        
        if close_side == 'SELL':
            limit_price = current_price * (1 - price_adjustment)  # Lower than market for selling
        else:
            limit_price = current_price * (1 + price_adjustment)  # Higher than market for buying
            
        # Get price precision
        price_precision = get_price_precision(binance_symbol)
        
        # Round limit price to proper precision
        limit_price = round(limit_price, price_precision)
        
        # Execute close order
        close_quantity = abs(position_amt)
        quantity_precision = get_quantity_precision(binance_symbol)
        close_quantity = round(close_quantity, quantity_precision)
        
        close_order = binance_client.futures_create_order(
            symbol=binance_symbol,
            type='LIMIT',
            side=close_side,
            quantity=close_quantity,
            price=limit_price,
            timeInForce='GTC',
            reduceOnly='true'
        )
        
        print(f"Position close order placed: {close_order}")
        
        # Wait for order to fill
        order_id = close_order['orderId']
        filled = False
        retries = 10
        
        while not filled and retries > 0:
            order_info = binance_client.futures_get_order(
                symbol=binance_symbol, 
                orderId=order_id
            )
            
            if order_info['status'] == 'FILLED':
                filled = True
                print("Close position order filled!")
            else:
                print(f"Waiting for close order to fill. Status: {order_info['status']}")
                time.sleep(1)
                retries -= 1
        
        # If not filled after retries, cancel and try market order as fallback
        if not filled:
            print(f"Close order not filled after {retries} retries. Cancelling and trying market order...")
            binance_client.futures_cancel_order(
                symbol=binance_symbol,
                orderId=order_id
            )
            
            # Fallback to market order to ensure position is closed
            market_close_order = binance_client.futures_create_order(
                symbol=binance_symbol,
                type='MARKET',
                side=close_side,
                quantity=close_quantity,
                reduceOnly='true'
            )
            
            print(f"Market close order executed: {market_close_order}")
            order_info = market_close_order
        
        # Get balance after closing
        balance_after = float(binance_client.futures_account_balance()[0]['balance'])
        
        return {
            'time': datetime.now(),
            'direction': 'close_' + current_direction,
            'exit_price': float(order_info.get('avgPrice', 0)) or float(order_info.get('price', 0)),
            'qty': close_quantity,
            'balance_after': balance_after
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to close position: {e}")
        return None

# Analyze market and make trade decision
def analyze_and_trade(row):
    global current_position, open_trade
    
    features = row[['ema_fast', 'ema_slow', 'rsi', 'atr']].values.reshape(1, -1)
    prob = clf.predict_proba(features)[0][1]
    
    # Check if we have a strong signal (>70% confidence)
    if prob < 0.7:
        return None
        
    price = row['close']
    direction = 'buy' if row['ema_fast'] > row['ema_slow'] else 'sell'
    
    # Calculate TP and SL levels
    tp_pct = tp_model.predict(features)[0] if hasattr(tp_model, 'predict') else 0.002
    sl_pct = sl_model.predict(features)[0] if hasattr(sl_model, 'predict') else 0.001
    
    tp_price = price * (1 + tp_pct) if direction == 'buy' else price * (1 - tp_pct)
    sl_price = price * (1 - sl_pct) if direction == 'buy' else price * (1 + sl_pct)
    
    # Check if we need to close opposite position first
    binance_position = get_current_position()
    
    if binance_position and binance_position != direction:
        print(f"Closing opposite position: {binance_position}")
        close_result = close_position(binance_position)
        if close_result:
            print(f"Position closed at {close_result['exit_price']}")
        
    # Execute new trade
    trade_result = execute_trade(direction, price, quantity)
    
    if trade_result:
        trade_data = {
            'time': row.name,
            'symbol': binance_symbol,
            'direction': direction,
            'entry': trade_result['entry'],
            'tp': tp_price,
            'sl': sl_price,
            'qty': trade_result['qty'],
            'confidence': prob
        }
        
        print(f"[TRADE] OPENED: {trade_data}")
        log.append(trade_data)
        pd.DataFrame(log).to_csv("trade_log.csv", index=False)
        current_position = direction
        open_trade = trade_data
        return trade_data
        
    return None

# Check if TP or SL is hit
def check_trade_exit(row):
    global current_position, open_trade, log
    
    if not open_trade:
        return
        
    price = row['close']
    direction = open_trade['direction']
    entry_price = open_trade['entry']
    tp_price = open_trade['tp']
    sl_price = open_trade['sl']
    exit_reason = None
    
    # Check if TP or SL levels hit
    if direction == 'buy':
        if price >= tp_price:
            exit_reason = "TP"
        elif price <= sl_price:
            exit_reason = "SL"
    elif direction == 'sell':
        if price <= tp_price:
            exit_reason = "TP"
        elif price >= sl_price:
            exit_reason = "SL"
            
    if exit_reason:
        # Close position
        close_result = close_position(direction)
        
        if close_result:
            # Calculate P&L
            pnl = close_result['balance_after'] - open_trade.get('balance_before', 0)
            
            closed_trade = {
                **open_trade,
                'exit_price': price,
                'exit_time': row.name,
                'pnl': round(pnl, 2),
                'exit_reason': exit_reason
            }
            
            print(f"[TRADE CLOSED] {exit_reason} at {price} | P&L: {round(pnl, 2)}")
            log.append(closed_trade)
            pd.DataFrame(log).to_csv("trade_log.csv", index=False)
            
            current_position = None
            open_trade = None

# Email summary function
def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'your@email.com'
    msg['To'] = 'your@email.com'
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('your@email.com', 'your_password')
            server.send_message(msg)
    except Exception as e:
        print(f"[EMAIL] Failed to send summary email: {e}")

# Main loop
def main():
    global current_position, open_trade
    
    print("[START] Enhanced AI Scalping Bot with Binance Futures API - Limit Order Version")
    
    # Set leverage and margin type
    set_leverage_and_margin_type()
    
    # Check if we have any existing position
    current_position = get_current_position()
    if current_position:
        print(f"[STARTUP] Detected existing {current_position} position")
    
    try:
        while True:
            # Fetch and process market data
            df = fetch_data()
            df = compute_indicators(df)
            df = label_data(df)
            
            # Train models periodically (can be adjusted to train less frequently)
            train_models(df)
            
            # Get latest data point for analysis
            row = df.iloc[-1]
            
            # First check if we need to exit any position
            check_trade_exit(row)
            
            # Then check if we should enter a new position
            if current_position is None:
                analyze_and_trade(row)
            
            # Sleep before next cycle
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("[STOP] Terminated by user")
        # Send summary email when stopped
        if log:
            summary = pd.DataFrame(log).to_string()
            send_email("AI Scalping Bot Trade Summary", summary)
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        # Send error notification
        send_email("AI Scalping Bot Error", f"Bot stopped due to error: {str(e)}")

if __name__ == "__main__":
    main()