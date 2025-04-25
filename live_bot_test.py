import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ========== CONFIG ==========
file_path = "WIFUSD_PERP-1m-2025-03.csv"  # Ensure this CSV file is in the correct directory
quantity = 100
retrain_every = 150  # Retrain every 150 candles to adapt to market changes

# Binance fees configuration
taker_fee_rate = 0.0005  # 0.05% taker fee (Standard Binance USDT-M futures fee)
# If you have VIP levels or BNB discount, adjust this value accordingly

# Hyperparameters
confidence_threshold = 0.75  # Entry threshold probability
tp_factor = 2.0  # Multiply ATR for take profit
sl_factor = 1.5  # Multiply ATR for stop loss
tsl_factor = 0.5  # Trailing stop loss factor

# Fee-adjusted minimum profit factor
# Ensure profits cover at least twice the round-trip commission
min_profit_fee_multiplier = 2.5  # Minimum profit must be 2.5x the round-trip commission

# ========== INIT ==========
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ========== MODEL ==========
def load_or_initialize_models():
    # Random Forest with better params for financial data
    rf_clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    
    # XGBoost with better params for imbalanced data
    xgb_clf = XGBClassifier(
        eval_metric='auc',
        learning_rate=0.05,
        n_estimators=200,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        gamma=1,
        random_state=42
    )
    
    # LightGBM optimized - fixed to prevent warnings
    lgbm_clf = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=32,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        # Fix for "No further splits" warnings
        min_child_samples=20,
        min_split_gain=0.01,
        random_state=42,
        verbosity=-1  # Suppress warnings
    )
    
    # Regressors for TP/SL with better hyperparameters
    tp_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
    
    sl_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42
    )
    
    # Feature scaler for better model performance
    scaler = StandardScaler()
    
    return rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler

rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler = load_or_initialize_models()

# ========== INDICATORS ==========
def compute_indicators(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Basic indicators
    df['ema_fast'] = df['close'].ewm(span=8).mean()  # Faster EMA for scalping
    df['ema_slow'] = df['close'].ewm(span=21).mean()
    df['ema_trend'] = df['close'].ewm(span=50).mean()  # Longer EMA for trend
    
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Enhanced ATR calculation - more responsive to volatility
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(10).mean()  # Shorter period for scalping
    
    # Enhanced MACD for faster signals
    df['macd'] = df['close'].ewm(span=8).mean() - df['close'].ewm(span=17).mean()
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    
    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']  # Volatility indicator
    
    # Volume analysis
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price action features
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Momentum indicators
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rate_of_change'] = df['close'].pct_change(5) * 100
    
    # Slope indicators for trend strength
    df['ema_fast_slope'] = df['ema_fast'].diff(3) / 3
    df['ema_slow_slope'] = df['ema_slow'].diff(3) / 3
    
    return df.dropna()

def label_data(df):
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Create more intelligent labels based on risk-reward
    # Looking for higher probability scalping setups
    lookahead_period = 5  # 5 minutes lookahead for scalping
    
    # Calculate future price movements
    df['future_high'] = df['high'].shift(-lookahead_period).rolling(lookahead_period).max()
    df['future_low'] = df['low'].shift(-lookahead_period).rolling(lookahead_period).min()
    df['future_close'] = df['close'].shift(-lookahead_period)
    
    # Calculate potential reward and risk
    df['potential_gain'] = (df['future_high'] - df['close']) / df['close']
    df['potential_loss'] = (df['close'] - df['future_low']) / df['close']
    
    # Calculate risk-reward ratio
    # Avoid division by zero
    df['potential_loss'] = df['potential_loss'].replace(0, 0.0001)
    df['risk_reward'] = df['potential_gain'] / df['potential_loss']
    
    # Account for fees when labeling - only label as good trades those that would be profitable after fees
    # Round-trip fee cost (entry + exit)
    round_trip_fee = 2 * taker_fee_rate
    
    # Label trades with good risk-reward and minimum price movement
    # Minimum move must be greater than round_trip_fee * min_profit_fee_multiplier
    min_move_pct = max(0.002, round_trip_fee * min_profit_fee_multiplier)  # 0.2% or fee-adjusted min
    min_rr_ratio = 1.5  # Minimum risk-reward ratio
    
    df['label'] = ((df['potential_gain'] > min_move_pct) & 
                   (df['risk_reward'] > min_rr_ratio)).astype(int)
    
    return df.dropna()

# ========== FEATURE ENGINEERING ==========
def create_feature_set(df):
    # Create more advanced features and indicators
    # Always create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Indicator crossovers (1 for cross up, -1 for cross down, 0 for no cross)
    df.loc[:, 'ema_cross'] = np.where(
        (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)), 
        1, np.where(
            (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)), 
            -1, 0)
    )
    
    df.loc[:, 'macd_cross'] = np.where(
        (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1)), 
        1, np.where(
            (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1)), 
            -1, 0)
    )
    
    # RSI categories
    df.loc[:, 'rsi_cat'] = np.where(df['rsi'] < 30, -2, 
                    np.where(df['rsi'] < 45, -1,
                    np.where(df['rsi'] > 70, 2,
                    np.where(df['rsi'] > 55, 1, 0))))
    
    # Price position relative to EMAs
    df.loc[:, 'price_vs_ema_fast'] = (df['close'] - df['ema_fast']) / df['close']
    df.loc[:, 'price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['close']
    df.loc[:, 'price_vs_ema_trend'] = (df['close'] - df['ema_trend']) / df['close']
    
    # Bollinger band position
    df.loc[:, 'bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR percent of price
    df.loc[:, 'atr_pct'] = df['atr'] / df['close']
    
    # Trend strength features
    df.loc[:, 'trend_strength'] = df['ema_fast_slope'] / df['atr_pct'].replace(0, 0.0001)
    
    return df

# ========== TRAIN ==========
def train_models(train_df):
    # First create enhanced features
    train_df = create_feature_set(train_df)
    
    # Define features for model training
    features = [
        'ema_fast', 'ema_slow', 'ema_trend', 'rsi', 'atr', 
        'macd', 'signal', 'macd_hist', 'bb_mid', 'bb_upper', 'bb_lower', 'bb_width',
        'volume_ratio', 'price_change', 'price_change_abs', 'high_low_ratio',
        'momentum', 'rate_of_change', 'ema_fast_slope', 'ema_slow_slope',
        'ema_cross', 'macd_cross', 'rsi_cat', 
        'price_vs_ema_fast', 'price_vs_ema_slow', 'price_vs_ema_trend',
        'bb_position', 'atr_pct', 'trend_strength'
    ]
    
    # Check which features are actually available in the dataframe
    available_features = [f for f in features if f in train_df.columns]
    
    # Handle NaN values - they can cause training failures
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    train_df = train_df.dropna(subset=available_features + ['label', 'potential_gain', 'potential_loss'])
    
    X = train_df[available_features]
    y = train_df['label'].values
    
    # Only train if we have enough samples
    if len(X) < 100 or sum(y) < 10 or sum(y) > len(y) - 10:
        print(f"[TRAIN] Warning: Not enough balanced data for training. Samples: {len(X)}, Positive: {sum(y)}")
        return available_features
    
    # Scale features for better model performance
    X_scaled = scaler.fit_transform(X)
    
    # Train classifiers
    print(f"[TRAIN] Training models with {len(X)} samples, {sum(y)} positive cases ({sum(y)/len(y)*100:.2f}%)")
    
    try:
        rf_clf.fit(X_scaled, y)
        xgb_clf.fit(X_scaled, y)
        lgbm_clf.fit(X, y)  # LightGBM handles feature names
        
        # Train TP/SL models
        # For TP, we want to predict the maximum possible gain in next few candles
        target_tp = train_df['potential_gain'].values
        # For SL, we want to predict the maximum possible loss to avoid
        target_sl = train_df['potential_loss'].values
        
        tp_model.fit(X_scaled, target_tp)
        sl_model.fit(X_scaled, target_sl)
    except Exception as e:
        print(f"[TRAIN] Error during model training: {e}")
    
    # Return the features list for prediction
    return available_features

# Ensemble prediction with weighted voting
def predict_proba(X_df, features):
    # Handle missing or inf values
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.mean())
    
    # Scale the input features
    X_scaled = scaler.transform(X_df[features])
    
    try:
        # Get probabilities from each classifier
        rf_proba = rf_clf.predict_proba(X_scaled)
        xgb_proba = xgb_clf.predict_proba(X_scaled)
        lgbm_proba = lgbm_clf.predict_proba(X_df[features])  # LightGBM gets DataFrame with feature names
        
        # Weighted average for ensemble (give more weight to XGBoost and LightGBM)
        weights = np.array([0.2, 0.4, 0.4])  # RF, XGB, LGBM
        avg_proba = (weights[0] * rf_proba + weights[1] * xgb_proba + weights[2] * lgbm_proba) / np.sum(weights)
        return avg_proba
    except Exception as e:
        print(f"[PREDICT] Error during prediction: {e}")
        # Return a default "no trade" probability
        return np.array([[1.0, 0.0]])

# ========== MARKET CONDITIONS ==========
def analyze_market_conditions(df, current_index):
    """Determine if current market conditions are favorable for trading"""
    # Safety check for index bounds
    if current_index < 10:
        return False, 'neutral'
    
    current_row = df.iloc[current_index]
    
    # Check for extreme volatility (avoid trading during high volatility)
    recent_bb_width = df['bb_width'].iloc[max(0, current_index-10):current_index].mean()
    current_bb_width = current_row['bb_width']
    high_volatility = current_bb_width > (recent_bb_width * 1.5)
    
    # Check for clear trend direction
    trend_alignment = ((current_row['ema_fast'] > current_row['ema_slow']) and 
                       (current_row['ema_slow'] > current_row['ema_trend'])) or \
                      ((current_row['ema_fast'] < current_row['ema_slow']) and 
                       (current_row['ema_slow'] < current_row['ema_trend']))
    
    # Check for sufficient volume
    sufficient_volume = current_row['volume_ratio'] > 0.8
    
    # Overall market condition assessment
    good_market_condition = (not high_volatility) and sufficient_volume
    
    # Direction bias based on trend
    if current_row['ema_fast'] > current_row['ema_slow'] and current_row['ema_slow_slope'] > 0:
        direction_bias = 'buy'
    elif current_row['ema_fast'] < current_row['ema_slow'] and current_row['ema_slow_slope'] < 0:
        direction_bias = 'sell'
    else:
        direction_bias = 'neutral'
        
    return good_market_condition, direction_bias

# ========== TRADE ==========
def calculate_fee(price, quantity):
    """Calculate the fee for a single trade (entry or exit)"""
    notional_value = price * quantity
    fee = notional_value * taker_fee_rate
    return fee

def simulate_trade(row, row_idx, df, current_position, open_trade, features):
    # Safety check
    if row_idx < 10 or row_idx >= len(df):
        return current_position, open_trade, None
    
    try:
        # First check market conditions
        good_market, direction_bias = analyze_market_conditions(df, row_idx)
        
        if not good_market:
            return current_position, open_trade, None
        
        # Create feature set for the current row
        # Get a single row as a DataFrame
        row_df = pd.DataFrame([row]).reset_index(drop=True)
        row_features_df = create_feature_set(row_df)
        
        # Create a DataFrame with just the required features
        X_df = pd.DataFrame(row_features_df[features].iloc[0]).T
        
        # Get prediction probabilities
        proba = predict_proba(X_df, features)[0]
        
        # Only take high confidence trades
        if proba[1] < confidence_threshold:
            return current_position, open_trade, None
        
        # Determine trade direction based on multiple factors
        if direction_bias != 'neutral':
            # Use the market bias direction if it's clear
            direction = direction_bias
        else:
            # Otherwise use technical indicators
            ema_signal = 'buy' if row['ema_fast'] > row['ema_slow'] else 'sell'
            macd_signal = 'buy' if row['macd'] > row['signal'] else 'sell'
            rsi_signal = 'buy' if row['rsi'] < 45 else 'sell' if row['rsi'] > 55 else 'neutral'
            
            # Count signals
            buy_signals = sum([1 for s in [ema_signal, macd_signal, rsi_signal] if s == 'buy'])
            sell_signals = sum([1 for s in [ema_signal, macd_signal, rsi_signal] if s == 'sell'])
            
            # Decide direction based on majority
            direction = 'buy' if buy_signals > sell_signals else 'sell'
        
        # Set entry price (account for slippage with market orders)
        entry = row['close']
        entry_fee = calculate_fee(entry, quantity)
        
        # Dynamic TP/SL based on volatility (ATR)
        # Scale predicted values by ATR for more realistic targets
        X_scaled = scaler.transform(X_df[features])
        
        try:
            predicted_tp_pct = tp_model.predict(X_scaled)[0]
            predicted_sl_pct = sl_model.predict(X_scaled)[0]
        except Exception as e:
            print(f"[TRADE] Error predicting TP/SL: {e}")
            predicted_tp_pct = row['atr'] / entry * 2
            predicted_sl_pct = row['atr'] / entry
        
        # Calculate round-trip fee cost as percentage of price
        round_trip_fee_pct = 2 * taker_fee_rate  # Entry + exit
        
        # Adjust TP to account for fees - ensure minimal profit covers fees with a margin
        # Minimum TP must cover at least min_profit_fee_multiplier times the round trip fee
        min_tp_pct = round_trip_fee_pct * min_profit_fee_multiplier
        
        # Scale by ATR and factors, ensuring it's at least covering fees
        tp_distance = max(predicted_tp_pct, row['atr'] / entry * tp_factor, min_tp_pct)
        sl_distance = max(predicted_sl_pct, row['atr'] / entry * sl_factor)
        
        # Set TP/SL levels
        tp = entry * (1 + tp_distance) if direction == 'buy' else entry * (1 - tp_distance)
        sl = entry * (1 - sl_distance) if direction == 'buy' else entry * (1 + sl_distance)
        
        # Create trade record
        open_trade = {
            'time': row.name,
            'entry': entry,
            'direction': direction,
            'tp': tp,
            'sl': sl,
            'entry_fee': entry_fee,
            'tsl_pct': row['atr'] / entry * tsl_factor,  # Dynamic trailing stop based on ATR
            'confidence': proba[1],
            'quantity': quantity
        }
        
        return direction, open_trade, {"entry": entry, "time": row.name, "dir": direction, "conf": f"{proba[1]:.2f}"}
    except Exception as e:
        print(f"[TRADE] Error in simulate_trade: {e}")
        return current_position, open_trade, None

def check_trade_exit(row, open_trade):
    try:
        price = row['close']
        entry = open_trade['entry']
        tp = open_trade['tp']
        sl = open_trade['sl']
        direction = open_trade['direction']
        tsl_pct = open_trade['tsl_pct']
        quantity = open_trade['quantity']
        exit_type = None

        # Calculate current profit/loss ratio
        if direction == 'buy':
            current_pnl_pct = (price - entry) / entry
        else:  # sell
            current_pnl_pct = (entry - price) / entry
        
        # Enhanced exit logic
        if direction == 'buy':
            if price >= tp:
                exit_type = 'TP'
            elif price <= sl:
                exit_type = 'SL'
            # More aggressive trailing stop when in profit
            elif price > entry:
                pnl_ratio = (price - entry) / (tp - entry) if tp != entry else 0  # How close to TP
                # The closer to TP, the tighter the trailing stop
                adaptive_tsl = tsl_pct * (1 - pnl_ratio * 0.5)  # Tighten as we approach TP
                new_sl = price * (1 - adaptive_tsl)
                if new_sl > sl:
                    open_trade['sl'] = new_sl
        else:  # sell direction
            if price <= tp:
                exit_type = 'TP'
            elif price >= sl:
                exit_type = 'SL'
            # More aggressive trailing stop when in profit
            elif price < entry:
                pnl_ratio = (entry - price) / (entry - tp) if entry != tp else 0  # How close to TP
                # The closer to TP, the tighter the trailing stop
                adaptive_tsl = tsl_pct * (1 - pnl_ratio * 0.5)  # Tighten as we approach TP
                new_sl = price * (1 + adaptive_tsl)
                if new_sl < sl:
                    open_trade['sl'] = new_sl

        if exit_type:
            # Calculate fees
            entry_fee = open_trade['entry_fee']
            exit_fee = calculate_fee(price, quantity)
            total_fees = entry_fee + exit_fee
            
            # Calculate raw P&L before fees
            raw_pnl = (price - entry) * quantity if direction == 'buy' else (entry - price) * quantity
            
            # Calculate net P&L after fees
            net_pnl = raw_pnl - total_fees
            
            return None, None, {
                "exit": exit_type,
                "exit_price": price,
                "exit_time": row.name,
                "raw_pnl": raw_pnl,
                "fees": total_fees,
                "net_pnl": net_pnl
            }
        return direction, open_trade, None
    except Exception as e:
        print(f"[EXIT] Error in check_trade_exit: {e}")
        # If there's an error during exit check, force exit to be safe
        try:
            price = row['close']
            entry = open_trade['entry']
            direction = open_trade['direction']
            quantity = open_trade['quantity']
            
            # Calculate fees
            entry_fee = open_trade['entry_fee']
            exit_fee = calculate_fee(price, quantity)
            total_fees = entry_fee + exit_fee
            
            # Calculate raw P&L before fees
            raw_pnl = (price - entry) * quantity if direction == 'buy' else (entry - price) * quantity
            
            # Calculate net P&L after fees
            net_pnl = raw_pnl - total_fees
            
            return None, None, {
                "exit": "ERROR",
                "exit_price": price,
                "exit_time": row.name,
                "raw_pnl": raw_pnl,
                "fees": total_fees,
                "net_pnl": net_pnl
            }
        except:
            return None, None, None

# ========== BACKTEST ==========
def run_backtest():
    print("[BACKTEST] Loading and preprocessing data...")
    try:
        df = pd.read_csv(file_path, parse_dates=['open_time'])
        df.set_index('open_time', inplace=True)
        df = compute_indicators(df)
        df = label_data(df)
    except Exception as e:
        print(f"[BACKTEST] Error during data loading: {e}")
        return
    
    # Initial training
    print("[BACKTEST] Performing initial model training...")
    train_size = min(1000, len(df) - 200)  # Ensure we have enough data for both training and testing
    available_features = train_models(df.iloc[:train_size])
    
    # Initialize backtest variables
    log = []
    current_position = None
    open_trade = None
    
    # Trading statistics
    trade_count = 0
    win_count = 0
    loss_count = 0
    total_raw_pnl = 0
    total_fees = 0
    total_net_pnl = 0
    max_drawdown = 0
    peak_capital = 0
    equity_curve = []
    
    print("[BACKTEST] Starting trading simulation...")
    
    for i in range(train_size, len(df)):
        if i % 1000 == 0:
            print(f"[BACKTEST] Processing candle {i}/{len(df)}, Net PnL: {total_net_pnl:.2f}, Fees: {total_fees:.2f}")
            
        row = df.iloc[i]
        
        # No open position - look for entry
        if current_position is None:
            current_position, open_trade, entry_log = simulate_trade(row, i, df, current_position, open_trade, available_features)
            if entry_log:
                log.append({**entry_log, "type": "entry"})
                trade_count += 1
        # Have open position - check for exit
        else:
            current_position, open_trade, exit_log = check_trade_exit(row, open_trade)
            if exit_log:
                log.append({**exit_log, "type": "exit"})
                
                # Update statistics
                raw_pnl = exit_log["raw_pnl"]
                fees = exit_log["fees"]
                net_pnl = exit_log["net_pnl"]
                
                total_raw_pnl += raw_pnl
                total_fees += fees
                total_net_pnl += net_pnl
                
                if net_pnl > 0:
                    win_count += 1
                else:
                    loss_count += 1
                    
                # Track equity curve and drawdown
                equity_curve.append(total_net_pnl)
                peak_capital = max(peak_capital, total_net_pnl)
                current_drawdown = peak_capital - total_net_pnl
                max_drawdown = max(max_drawdown, current_drawdown)
        
        # Retrain models periodically to adapt to changing market conditions
        if retrain_every > 0 and i % retrain_every == 0 and i > train_size:
            print(f"[BACKTEST] Retraining models at index {i}/{len(df)}, current Net P&L: {total_net_pnl:.2f}")
            try:
                # Use recent data with a sliding window approach
                window_start = max(0, i - 2000)  # Use last 2000 candles at most
                available_features = train_models(df.iloc[window_start:i])
            except Exception as e:
                print(f"[BACKTEST] Error during retraining: {e}")
    
    # ========== ANALYZE RESULTS ==========
    if log:
        try:
            log_df = pd.DataFrame(log)
            log_df.to_csv("logs/backtest_results.csv", index=False)
            
            # Calculate statistics
            if not log_df.empty:
                entry_logs = log_df[log_df['type'] == 'entry']
                exit_logs = log_df[log_df['type'] == 'exit']
                
                # Continue from previous code
                if not exit_logs.empty and 'net_pnl' in exit_logs.columns:
                    win_rate = win_count / trade_count if trade_count > 0 else 0
                    avg_win = exit_logs[exit_logs['net_pnl'] > 0]['net_pnl'].mean() if win_count > 0 else 0
                    avg_loss = exit_logs[exit_logs['net_pnl'] < 0]['net_pnl'].mean() if loss_count > 0 else 0
                    profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if loss_count > 0 and avg_loss != 0 else float('inf')
                    
                    # Calculate fee impact
                    fee_impact_pct = (total_fees / total_raw_pnl * 100) if total_raw_pnl > 0 else 0
                    
                    print("\n[BACKTEST] Results Summary:")
                    print(f"Total Raw P&L: {total_raw_pnl:.2f}")
                    print(f"Total Fees: {total_fees:.2f} ({fee_impact_pct:.2f}% of raw P&L)")
                    print(f"Total Net P&L: {total_net_pnl:.2f}")
                    print(f"Total Trades: {trade_count}")
                    print(f"Win Rate: {win_rate:.2%}")
                    print(f"Average Win: {avg_win:.2f}")
                    print(f"Average Loss: {avg_loss:.2f}")
                    print(f"Profit Factor: {profit_factor:.2f}")
                    print(f"Maximum Drawdown: {max_drawdown:.2f}")
                    
                    # ROI calculation
                    # Assuming initial capital based on position size
                    initial_capital = quantity * df.iloc[train_size]['close']
                    roi = (total_net_pnl / initial_capital) * 100 if initial_capital > 0 else 0
                    print(f"ROI: {roi:.2f}%")
                    
                    # Save detailed statistics to file
                    stats = {
                        'total_raw_pnl': total_raw_pnl,
                        'total_fees': total_fees,
                        'total_net_pnl': total_net_pnl,
                        'fee_impact_pct': fee_impact_pct,
                        'trade_count': trade_count,
                        'win_count': win_count,
                        'loss_count': loss_count,
                        'win_rate': win_rate,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': profit_factor,
                        'max_drawdown': max_drawdown,
                        'roi': roi,
                        'initial_capital': initial_capital
                    }
                    
                    pd.DataFrame([stats]).to_csv("logs/backtest_stats.csv", index=False)
                    
                    # Save equity curve
                    pd.DataFrame({'equity': equity_curve}).to_csv("logs/equity_curve.csv", index=False)
                    
                    # Generate fee analysis
                    fee_analysis = {
                        'avg_fee_per_trade': total_fees / trade_count if trade_count > 0 else 0,
                        'fee_to_pnl_ratio': total_fees / total_raw_pnl if total_raw_pnl > 0 else float('inf'),
                        'net_profit_per_trade': total_net_pnl / trade_count if trade_count > 0 else 0
                    }
                    
                    pd.DataFrame([fee_analysis]).to_csv("logs/fee_analysis.csv", index=False)
                    
                    # Create trade performance breakdown by hour of day
                    if 'exit_time' in exit_logs.columns:
                        exit_logs['hour'] = pd.to_datetime(exit_logs['exit_time']).dt.hour
                        hourly_performance = exit_logs.groupby('hour')['net_pnl'].agg(
                            ['sum', 'mean', 'count']).reset_index()
                        hourly_performance.to_csv("logs/hourly_performance.csv", index=False)
        except Exception as e:
            print(f"[BACKTEST] Error during results analysis: {e}")
    else:
        print("[BACKTEST] No trades were executed.")
    
    print("[BACKTEST] Done. Results logged to logs directory.")

if __name__ == "__main__":
    run_backtest()