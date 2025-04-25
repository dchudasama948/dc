import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
import optuna
from functools import partial
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# ========== CONFIG ==========
file_path = "WIFUSD_PERP-1m-2025-03.csv"  # Ensure this CSV file is in the correct directory
initial_balance = 10000  # Starting account balance
risk_per_trade = 0.01    # Risk 1% of account per trade
retrain_every = 120      # Retrain models more frequently (120 instead of 150)

# Hyperparameters - improved values
confidence_threshold = 0.85  # Increased from 0.75 for stronger signals
tp_factor = 1.5  # Reduced from 2.0 for faster profits
sl_factor = 1.0  # Tighter stop loss
tsl_factor = 0.3  # More aggressive trailing stop
min_signal_strength = 50  # Minimum signal strength score to enter trade

# ========== INIT ==========
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("analysis", exist_ok=True)

# ========== MODEL ==========
def load_or_initialize_models():
    # Random Forest with better params for financial data
    rf_clf = RandomForestClassifier(
        n_estimators=300,  # Increased from 200
        max_depth=8,
        min_samples_leaf=15,  # Increased from 10 to reduce overfitting
        class_weight='balanced_subsample',  # Changed to balanced_subsample
        random_state=42,
        max_features='sqrt',  # Added to reduce overfitting
        bootstrap=True,
        n_jobs=-1  # Use all available cores
    )
    
    # XGBoost with better params for imbalanced data
    xgb_clf = XGBClassifier(
        eval_metric='auc',
        learning_rate=0.03,  # Reduced from 0.05 for better generalization
        n_estimators=300,    # Increased from 200
        max_depth=5,         # Reduced from 6 to prevent overfitting
        subsample=0.7,       # Changed from 0.8
        colsample_bytree=0.7,  # Changed from 0.8
        scale_pos_weight=5,    # Increased from 3 for more imbalanced data
        gamma=1.5,             # Increased from 1
        min_child_weight=5,    # Added to control overfitting
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.2,        # L2 regularization
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # LightGBM optimized
    lgbm_clf = LGBMClassifier(
        n_estimators=300,        # Increased from 200
        learning_rate=0.03,      # Reduced from 0.05
        max_depth=5,             # Reduced from 6
        num_leaves=24,           # Reduced from 32
        subsample=0.7,           # Changed from 0.8
        colsample_bytree=0.7,    # Changed from 0.8
        scale_pos_weight=5,      # Increased from 3
        min_child_samples=25,    # Increased from 20
        min_split_gain=0.02,     # Increased from 0.01
        reg_alpha=0.1,           # Added L1 regularization
        reg_lambda=1.2,          # Added L2 regularization
        random_state=42,
        verbosity=-1,            # Suppress warnings
        n_jobs=-1                # Use all available cores
    )
    
    # Regressors for TP/SL with better hyperparameters
    tp_model = RandomForestRegressor(
        n_estimators=200,        # Increased from 150
        max_depth=5,             # Reduced from 6
        min_samples_leaf=15,     # Increased from 10
        max_features='sqrt',     # Added feature restriction
        bootstrap=True,          # Explicit bootstrap
        random_state=42,
        n_jobs=-1                # Use all available cores
    )
    
    sl_model = RandomForestRegressor(
        n_estimators=200,        # Increased from 150
        max_depth=5,             # Reduced from 6
        min_samples_leaf=15,     # Increased from 10
        max_features='sqrt',     # Added feature restriction
        bootstrap=True,          # Explicit bootstrap
        random_state=42,
        n_jobs=-1                # Use all available cores
    )
    
    # Feature scaler for better model performance
    scaler = StandardScaler()
    
    return rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler

rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler = load_or_initialize_models()

# ========== INDICATORS AND TECHNICAL ANALYSIS ==========
def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)"""
    df = df.copy()
    
    # True Range
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()
    
    # Plus Directional Movement (+DM)
    df['plus_dm'] = 0.0
    df.loc[(df['high'] > df['high'].shift(1)) & 
           (df['high'] - df['high'].shift(1) > df['low'].shift(1) - df['low']), 'plus_dm'] = \
        df['high'] - df['high'].shift(1)
    
    # Minus Directional Movement (-DM)
    df['minus_dm'] = 0.0
    df.loc[(df['low'] < df['low'].shift(1)) & 
           (df['low'].shift(1) - df['low'] > df['high'] - df['high'].shift(1)), 'minus_dm'] = \
        df['low'].shift(1) - df['low']
    
    # Smooth the +DM and -DM with Wilder's smoothing
    df['smoothed_plus_dm'] = df['plus_dm'].rolling(period).sum()
    df['smoothed_minus_dm'] = df['minus_dm'].rolling(period).sum()
    
    # Calculate +DI and -DI
    df['plus_di'] = 100 * df['smoothed_plus_dm'] / df['atr'].replace(0, np.nan)
    df['minus_di'] = 100 * df['smoothed_minus_dm'] / df['atr'].replace(0, np.nan)
    
    # Calculate DX and ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan)
    df['adx'] = df['dx'].rolling(period).mean()
    
    return df['adx']

def compute_indicators(df):
    """Enhanced indicator calculation with additional features"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # === Basic indicators ===
    # Enhanced EMAs for different timeframes
    df['ema_very_fast'] = df['close'].ewm(span=5).mean()   # Very fast EMA for quick signals
    df['ema_fast'] = df['close'].ewm(span=8).mean()        # Fast EMA for scalping
    df['ema_medium'] = df['close'].ewm(span=13).mean()     # Medium EMA
    df['ema_slow'] = df['close'].ewm(span=21).mean()       # Slow EMA
    df['ema_trend'] = df['close'].ewm(span=50).mean()      # Longer EMA for trend
    df['ema_long'] = df['close'].ewm(span=100).mean()      # Long EMA for major trend
    
    # Simple moving averages for different perspectives
    df['sma_fast'] = df['close'].rolling(10).mean()
    df['sma_slow'] = df['close'].rolling(30).mean()
    
    # === RSI and RSI-based indicators ===
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()  # Using EMA instead of SMA for smoother RSI
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSI-based additional indicators
    df['rsi_ma'] = df['rsi'].rolling(5).mean()  # RSI moving average
    df['rsi_slope'] = df['rsi'].diff(3)         # RSI slope - momentum of RSI
    
    # === Enhanced volatility indicators ===
    # ATR calculation with multiple timeframes
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(10).mean()            # Short ATR for scalping
    df['atr_slow'] = tr.rolling(20).mean()       # Longer ATR for overall volatility
    df['atr_pct'] = df['atr'] / df['close'] * 100  # ATR as percentage of price
    
    # Bollinger Bands with multiple lookbacks
    for period in [10, 20]:
        df[f'bb_mid_{period}'] = df['close'].rolling(period).mean()
        df[f'bb_std_{period}'] = df['close'].rolling(period).std()
        df[f'bb_upper_{period}'] = df[f'bb_mid_{period}'] + 2 * df[f'bb_std_{period}']
        df[f'bb_lower_{period}'] = df[f'bb_mid_{period}'] - 2 * df[f'bb_std_{period}']
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_mid_{period}']
    
    # Use standard period for main BB features
    df['bb_mid'] = df['bb_mid_20']
    df['bb_upper'] = df['bb_upper_20']
    df['bb_lower'] = df['bb_lower_20']
    df['bb_width'] = df['bb_width_20']
    
    # Price in relation to Bollinger Bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-9)
    
    # === Enhanced MACD and derivatives ===
    # MACD with multiple parameter sets
    df['macd'] = df['close'].ewm(span=8).mean() - df['close'].ewm(span=17).mean()  # Fast MACD for scalping
    df['signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['signal']
    df['macd_hist_slope'] = df['macd_hist'].diff(3)  # Slope of MACD histogram
    
    # Additional slower MACD for trend confirmation
    df['macd_slow'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['signal_slow'] = df['macd_slow'].ewm(span=9).mean()
    df['macd_hist_slow'] = df['macd_slow'] - df['signal_slow']
    
    # === Volume analysis ===
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # OBV (On-Balance Volume)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma'] = df['obv'].rolling(20).mean()
    
    # === Price action features ===
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['price_change'].abs()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Candlestick patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Simple candlestick pattern identifiers
    df['hammer'] = ((df['lower_wick'] > 2 * df['body_size']) & 
                   (df['upper_wick'] < 0.5 * df['body_size']) & 
                   (df['body_size'] > 0.001)).astype(int)
    
    df['shooting_star'] = ((df['upper_wick'] > 2 * df['body_size']) & 
                          (df['lower_wick'] < 0.5 * df['body_size']) & 
                          (df['body_size'] > 0.001)).astype(int)
    
    df['doji'] = (df['body_size'] < 0.001).astype(int)
    
    # === Momentum indicators ===
    df['momentum'] = df['close'] - df['close'].shift(5)
    df['rate_of_change'] = df['close'].pct_change(5) * 100
    
    # === Trend strength indicators ===
    # Slope indicators for trend identification
    for period in [3, 5, 8]:
        df[f'ema_fast_slope_{period}'] = df['ema_fast'].diff(period) / period
        df[f'ema_slow_slope_{period}'] = df['ema_slow'].diff(period) / period
    
    # Main slope indicators
    df['ema_fast_slope'] = df['ema_fast_slope_3']
    df['ema_slow_slope'] = df['ema_slow_slope_3']
    
    # ADX for trend strength
    df['adx'] = calculate_adx(df, 14)
    
    # === Support/Resistance zones identification ===
    # Rolling min/max for S/R levels
    for period in [10, 20, 50]:
        df[f'rolling_high_{period}'] = df['high'].rolling(period).max()
        df[f'rolling_low_{period}'] = df['low'].rolling(period).min()
    
    # Distance from key levels
    df['dist_from_recent_high'] = (df['close'] - df['rolling_high_20']) / df['close']
    df['dist_from_recent_low'] = (df['close'] - df['rolling_low_20']) / df['close']
    
    # === Higher timeframe features ===
    # We'll aggregate to 5-minute data for trend confirmation
    # This requires resampling the data
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    try:
        # This requires the index to be a datetime
        if isinstance(df.index, pd.DatetimeIndex):
            # Resample to 5-minute timeframe
            df_5min = df.resample('5min').agg(ohlc_dict)
            
            # Calculate EMAs on 5-minute timeframe
            df_5min['ema_fast_5m'] = df_5min['close'].ewm(span=8).mean()
            df_5min['ema_slow_5m'] = df_5min['close'].ewm(span=21).mean()
            
            # Merge back into the original dataframe
            df['ema_fast_5m'] = df_5min['ema_fast_5m'].reindex(df.index, method='ffill')
            df['ema_slow_5m'] = df_5min['ema_slow_5m'].reindex(df.index, method='ffill')
            
            # Calculate 5-minute trend direction
            df['trend_5m'] = np.where(df['ema_fast_5m'] > df['ema_slow_5m'], 1, -1)
    except Exception as e:
        print(f"Error creating higher timeframe features: {e}")
        # Create dummy columns if resampling fails
        df['ema_fast_5m'] = df['ema_fast']
        df['ema_slow_5m'] = df['ema_slow']
        df['trend_5m'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    return df.dropna()

def label_data(df):
    """Enhanced labeling with more sophisticated risk-reward analysis"""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # We'll use multiple lookahead periods to find better setups
    # For scalping trades (very short-term)
    lookahead_period_short = 5   # 5 minutes for quick scalps
    # For slightly longer trades
    lookahead_period_medium = 10  # 10 minutes for standard trades
    
    # === Calculate future price movements ===
    # Short timeframe
    df['future_high_short'] = df['high'].shift(-lookahead_period_short).rolling(lookahead_period_short).max()
    df['future_low_short'] = df['low'].shift(-lookahead_period_short).rolling(lookahead_period_short).min()
    df['future_close_short'] = df['close'].shift(-lookahead_period_short)
    
    # Medium timeframe
    df['future_high_medium'] = df['high'].shift(-lookahead_period_medium).rolling(lookahead_period_medium).max()
    df['future_low_medium'] = df['low'].shift(-lookahead_period_medium).rolling(lookahead_period_medium).min()
    df['future_close_medium'] = df['close'].shift(-lookahead_period_medium)
    
    # === Calculate potential gains and losses ===
    # Potential gains (using max high in the lookahead period)
    df['potential_gain_short'] = (df['future_high_short'] - df['close']) / df['close']
    df['potential_gain_medium'] = (df['future_high_medium'] - df['close']) / df['close']
    
    # Potential losses (using min low in the lookahead period)
    df['potential_loss_short'] = (df['close'] - df['future_low_short']) / df['close']
    df['potential_loss_medium'] = (df['close'] - df['future_low_medium']) / df['close']
    
    # Avoid division by zero
    df['potential_loss_short'] = df['potential_loss_short'].replace(0, 0.0001)
    df['potential_loss_medium'] = df['potential_loss_medium'].replace(0, 0.0001)
    
    # Calculate risk-reward ratios
    df['risk_reward_short'] = df['potential_gain_short'] / df['potential_loss_short']
    df['risk_reward_medium'] = df['potential_gain_medium'] / df['potential_loss_medium']
    
    # === Create adaptive labels based on market conditions ===
    # Adjust minimum required movement based on ATR
    df['atr_based_min_move'] = df['atr_pct'] * 0.5  # 50% of ATR
    
    # Set minimum thresholds
    min_move_pct_floor = 0.0015  # 0.15% absolute minimum move
    min_rr_ratio = 1.8  # Minimum risk-reward ratio (increased)
    
    # Adaptive labeling based on volatility and potential reward
    # For short-term scalping label
    df['potential_move_vs_atr_short'] = df['potential_gain_short'] / df['atr_pct']
    
    # Label for short-term trades
    df['label_short'] = (
        (df['potential_gain_short'] > np.maximum(df['atr_based_min_move'], min_move_pct_floor)) & 
        (df['risk_reward_short'] > min_rr_ratio) &
        (df['potential_move_vs_atr_short'] > 0.8)  # Move should be significant compared to ATR
    ).astype(int)
    
    # Use the short-term label as our primary label
    df['label'] = df['label_short']
    
    # Also calculate potential gain and loss
    df['potential_gain'] = df['potential_gain_short']
    df['potential_loss'] = df['potential_loss_short']
    
    return df.dropna()

# ========== FEATURE ENGINEERING ==========
def create_feature_set(df):
    """Enhanced feature engineering with more advanced technical indicators"""
    # Always create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # === Indicator crossovers and relative positions ===
    # EMA crosses (1 for cross up, -1 for cross down, 0 for no cross)
    df.loc[:, 'ema_cross'] = np.where(
        (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)), 
        1, np.where(
            (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)), 
            -1, 0)
    )
    
    # MACD crosses
    df.loc[:, 'macd_cross'] = np.where(
        (df['macd'] > df['signal']) & (df['macd'].shift(1) <= df['signal'].shift(1)), 
        1, np.where(
            (df['macd'] < df['signal']) & (df['macd'].shift(1) >= df['signal'].shift(1)), 
            -1, 0)
    )
    
    # RSI categories for easier model interpretation
    df.loc[:, 'rsi_cat'] = np.where(df['rsi'] < 30, -2, 
                    np.where(df['rsi'] < 45, -1,
                    np.where(df['rsi'] > 70, 2,
                    np.where(df['rsi'] > 55, 1, 0))))
    
    # RSI crosses
    df.loc[:, 'rsi_cross_30'] = np.where(
        (df['rsi'] > 30) & (df['rsi'].shift(1) <= 30), 1, 
        np.where((df['rsi'] < 30) & (df['rsi'].shift(1) >= 30), -1, 0)
    )
    
    df.loc[:, 'rsi_cross_70'] = np.where(
        (df['rsi'] > 70) & (df['rsi'].shift(1) <= 70), 1, 
        np.where((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70), -1, 0)
    )
    
    # === Price position relative to EMAs and MAs ===
    df.loc[:, 'price_vs_ema_fast'] = (df['close'] - df['ema_fast']) / df['close']
    df.loc[:, 'price_vs_ema_slow'] = (df['close'] - df['ema_slow']) / df['close']
    df.loc[:, 'price_vs_ema_trend'] = (df['close'] - df['ema_trend']) / df['close']
    
    # === EMA relationships ===
    # EMA alignment - how well EMAs are aligned in a trend
    df.loc[:, 'ema_alignment'] = np.where(
        (df['ema_very_fast'] > df['ema_fast']) & 
        (df['ema_fast'] > df['ema_medium']) & 
        (df['ema_medium'] > df['ema_slow']) & 
        (df['ema_slow'] > df['ema_trend']), 
        1,  # Strong uptrend
        np.where(
            (df['ema_very_fast'] < df['ema_fast']) & 
            (df['ema_fast'] < df['ema_medium']) & 
            (df['ema_medium'] < df['ema_slow']) & 
            (df['ema_slow'] < df['ema_trend']),
            -1,  # Strong downtrend
            0   # No clear trend
        )
    )
    
    # === Volatility features ===
    df.loc[:, 'atr_pct'] = df['atr'] / df['close']
    df.loc[:, 'volatility_change'] = df['atr'].pct_change(5)
    df.loc[:, 'close_to_atr_ratio'] = df['close'] / df['atr']
    
    # === Advanced trend strength metrics ===
    df.loc[:, 'trend_strength'] = df['ema_fast_slope'] / (df['atr_pct'] + 1e-9)
    df.loc[:, 'adx_trend_strength'] = np.where(df['adx'] > 25, 1, 0)  # ADX > 25 indicates trend
    
    # Combine ADX with trend direction
    df.loc[:, 'adx_directional_trend'] = np.where(
        (df['adx'] > 25) & (df['ema_fast'] > df['ema_slow']), 
        1,  # Strong uptrend
        np.where(
            (df['adx'] > 25) & (df['ema_fast'] < df['ema_slow']), 
            -1,  # Strong downtrend
            0   # No strong trend
        )
    )
    
    # === Pattern sequence detection ===
    # Detect potential reversal patterns based on 3-candle sequences
    for i in range(3):
        df[f'close_up_{i+1}'] = (df['close'].shift(i) > df['open'].shift(i)).astype(int)
    
    # 3 consecutive bullish candles
    df.loc[:, 'three_bull_candles'] = ((df['close_up_1'] & df['close_up_2'] & df['close_up_3']).astype(int))
    # 3 consecutive bearish candles
    df.loc[:, 'three_bear_candles'] = ((~df['close_up_1'] & ~df['close_up_2'] & ~df['close_up_3']).astype(int))
    
    # === Mean reversion potential ===
    # RSI extremes
    df.loc[:, 'rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df.loc[:, 'rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    # Price distance from Bollinger bands
    df.loc[:, 'price_below_lower_band'] = (df['close'] < df['bb_lower']).astype(int)
    df.loc[:, 'price_above_upper_band'] = (df['close'] > df['bb_upper']).astype(int)
    
    # === Volume confirmation features ===
    # Volume increasing with price
    df.loc[:, 'bull_vol_confirm'] = ((df['close'] > df['open']) & (df['volume'] > df['volume'].shift(1))).astype(int)
    df.loc[:, 'bear_vol_confirm'] = ((df['close'] < df['open']) & (df['volume'] > df['volume'].shift(1))).astype(int)
    
    # === Higher timeframe trend alignment ===
    if 'trend_5m' in df.columns:
        df.loc[:, 'align_with_higher_tf'] = np.where(
            (df['ema_fast'] > df['ema_slow']) & (df['trend_5m'] == 1), 
            1,  # 1-min and 5-min both bullish
            np.where(
                (df['ema_fast'] < df['ema_slow']) & (df['trend_5m'] == -1), 
                -1,  # 1-min and 5-min both bearish
                0    # Misalignment between timeframes
            )
        )
    else:
        df.loc[:, 'align_with_higher_tf'] = 0
    
    # === Support/Resistance proximity ===
    # Check if price is near recent highs/lows
    df.loc[:, 'near_recent_high'] = (abs(df['close'] - df['rolling_high_20']) / df['close'] < 0.002).astype(int)
    df.loc[:, 'near_recent_low'] = (abs(df['close'] - df['rolling_low_20']) / df['close'] < 0.002).astype(int)
    
    return df

# ========== TIME FILTER ==========
def time_filter(timestamp):
    """Filter out trades during potentially unfavorable times"""
    # Convert to datetime if it's not already
    if not isinstance(timestamp, pd.Timestamp) and not isinstance(timestamp, datetime):
        try:
            timestamp = pd.to_datetime(timestamp)
        except:
            # If conversion fails, return True (don't filter)
            return True
    
    hour = timestamp.hour
    minute = timestamp.minute
    weekday = timestamp.weekday()
    
    # Avoid trading during low liquidity hours
    if 22 <= hour or hour < 2:
        return False
    
    # Avoid trading during potential news events or low liquidity
    # Friday end of day
    if weekday == 4 and hour >= 20:
        return False
        
    return True

# ========== MARKET CONDITIONS ==========
# Continuing from where the code left off...

def analyze_market_conditions(df, current_idx):
    """Analyze current market conditions for more informed trading decisions"""
    # Use a window of recent data for analysis
    window = 30
    start_idx = max(0, current_idx - window)
    recent_data = df.iloc[start_idx:current_idx+1]
    
    # Market condition indicators
    conditions = {
        'volatility': recent_data['atr_pct'].iloc[-1],
        'trend_strength': recent_data['adx'].iloc[-1] if 'adx' in recent_data else 0,
        'trend_direction': 1 if recent_data['ema_fast'].iloc[-1] > recent_data['ema_slow'].iloc[-1] else -1,
        'rsi_level': recent_data['rsi'].iloc[-1],
        'bb_width': recent_data['bb_width'].iloc[-1] if 'bb_width' in recent_data else 0,
        'volume_surge': recent_data['volume'].iloc[-1] / recent_data['volume'].iloc[:-1].mean() if len(recent_data) > 1 else 1
    }
    
    # Calculate market regime scores
    regime_scores = {
        'trending': min(100, conditions['trend_strength'] * 4),  # ADX-based trend strength
        'mean_reverting': min(100, max(0, (70 - conditions['trend_strength']) * 1.5)),  # Lower ADX favors mean reversion
        'volatile': min(100, conditions['volatility'] * 10000),  # ATR percentage-based volatility
        'quiet': min(100, max(0, (0.003 - conditions['volatility']) * 30000)),  # Low volatility periods
    }
    
    # Determine dominant regime
    dominant_regime = max(regime_scores, key=regime_scores.get)
    dominant_score = regime_scores[dominant_regime]
    
    # Direction bias based on multiple indicators
    direction_signals = {
        'ema': 1 if conditions['trend_direction'] > 0 else -1,
        'rsi': 1 if conditions['rsi_level'] < 40 else (-1 if conditions['rsi_level'] > 60 else 0),
        'bb': -1 if recent_data['bb_position'].iloc[-1] > 0.8 else (1 if recent_data['bb_position'].iloc[-1] < 0.2 else 0),
        'macd': 1 if recent_data['macd_hist'].iloc[-1] > 0 else -1
    }
    
    # Calculate overall direction score (-100 to 100)
    direction_weights = {'ema': 0.4, 'rsi': 0.2, 'bb': 0.2, 'macd': 0.2}
    direction_score = sum(direction_signals[k] * direction_weights[k] for k in direction_weights) * 100
    
    return {
        'regime': dominant_regime,
        'regime_score': dominant_score,
        'direction': 'bullish' if direction_score > 30 else ('bearish' if direction_score < -30 else 'neutral'),
        'direction_score': direction_score,
        'volatility': conditions['volatility'],
        'volume_surge': conditions['volume_surge'],
        'raw_scores': regime_scores
    }

# ========== SIGNAL GENERATION ==========
def generate_signal(models, features, current_market_conditions):
    """Generate trading signals based on model predictions and market conditions"""
    rf_clf, xgb_clf, lgbm_clf = models
    
    # Get predictions and probabilities from all models
    rf_pred = rf_clf.predict(features)
    rf_prob = rf_clf.predict_proba(features)
    
    xgb_pred = xgb_clf.predict(features)
    xgb_prob = xgb_clf.predict_proba(features)
    
    lgbm_pred = lgbm_clf.predict(features)
    lgbm_prob = lgbm_clf.predict_proba(features)
    
    # Extract probability of positive class (long signal)
    rf_long_prob = rf_prob[:, 1][0] if rf_prob.shape[1] > 1 else 0
    xgb_long_prob = xgb_prob[:, 1][0] if xgb_prob.shape[1] > 1 else 0
    lgbm_long_prob = lgbm_prob[:, 1][0] if lgbm_prob.shape[1] > 1 else 0
    
    # Weighted ensemble (adjusted weights)
    ensemble_weights = {
        'trending': {'rf': 0.3, 'xgb': 0.4, 'lgbm': 0.3},  # XGBoost works well in trends
        'mean_reverting': {'rf': 0.4, 'xgb': 0.3, 'lgbm': 0.3},  # Random Forest good for mean reversion
        'volatile': {'rf': 0.2, 'xgb': 0.5, 'lgbm': 0.3},  # XGBoost handles volatility well
        'quiet': {'rf': 0.4, 'xgb': 0.3, 'lgbm': 0.3}      # Random Forest more stable in quiet markets
    }
    
    # Select weights based on current market regime
    current_regime = current_market_conditions['regime']
    weights = ensemble_weights.get(current_regime, {'rf': 0.33, 'xgb': 0.33, 'lgbm': 0.34})
    
    # Calculate weighted probability
    ensemble_prob = (
        rf_long_prob * weights['rf'] +
        xgb_long_prob * weights['xgb'] +
        lgbm_long_prob * weights['lgbm']
    )
    
    # Adjust confidence threshold based on market conditions
    base_threshold = confidence_threshold  # From config
    
    # Adjust threshold based on regime
    threshold_adjustments = {
        'trending': -0.05,  # Lower threshold in trending markets for more signals
        'mean_reverting': 0.05,  # Higher threshold in mean reverting markets
        'volatile': 0.1,    # Much higher threshold in volatile markets
        'quiet': -0.02      # Slightly lower threshold in quiet markets
    }
    
    adjusted_threshold = base_threshold + threshold_adjustments.get(current_regime, 0)
    
    # Direction alignment adjustment - boost confidence if models align with market direction
    if (ensemble_prob > 0.5 and current_market_conditions['direction'] == 'bullish') or \
       (ensemble_prob < 0.5 and current_market_conditions['direction'] == 'bearish'):
        ensemble_prob += 0.05  # Boost confidence when aligned
    
    # Calculate signal strength (0-100)
    raw_strength = abs(ensemble_prob - 0.5) * 200  # Convert from 0.5-1.0 range to 0-100
    
    # Adjust strength based on market conditions
    strength_multiplier = 1.0
    
    # Stronger signals in trending markets, weaker in volatile
    if current_regime == 'trending' and current_market_conditions['regime_score'] > 60:
        strength_multiplier += 0.2
    elif current_regime == 'volatile' and current_market_conditions['regime_score'] > 70:
        strength_multiplier -= 0.3
    
    # Volume surge can increase signal strength
    if current_market_conditions['volume_surge'] > 1.5:
        strength_multiplier += 0.1
    
    signal_strength = min(100, raw_strength * strength_multiplier)
    
    # Determine signal type
    if ensemble_prob >= adjusted_threshold and signal_strength >= min_signal_strength:
        signal = 'buy'
    elif ensemble_prob <= (1 - adjusted_threshold) and signal_strength >= min_signal_strength:
        signal = 'sell'
    else:
        signal = 'neutral'
    
    return {
        'signal': signal, 
        'strength': signal_strength,
        'confidence': max(ensemble_prob, 1 - ensemble_prob),
        'regime': current_regime,
        'threshold': adjusted_threshold,
        'raw_ensemble_prob': ensemble_prob,
        'model_probs': {
            'rf': rf_long_prob,
            'xgb': xgb_long_prob,
            'lgbm': lgbm_long_prob
        }
    }

# ========== TRADE MANAGEMENT ==========
def calculate_position_size(balance, risk_per_trade, price, stop_loss_price):
    """Calculate position size based on account risk management"""
    if stop_loss_price == 0 or price == stop_loss_price:
        return 0
    
    # Calculate risk amount in account currency
    risk_amount = balance * risk_per_trade
    
    # Calculate risk per unit
    risk_per_unit = abs(price - stop_loss_price)
    
    # Calculate position size
    if risk_per_unit > 0:
        position_size = risk_amount / risk_per_unit
    else:
        position_size = 0
    
    return position_size

def calculate_take_profit_stop_loss(models, features, price, signal_type, market_conditions, current_volatility):
    """Calculate adaptive take profit and stop loss levels based on market conditions"""
    tp_model, sl_model = models
    
    # Base TP/SL on model predictions
    try:
        tp_prediction = tp_model.predict(features)[0]
        sl_prediction = sl_model.predict(features)[0]
    except:
        # Default if prediction fails
        tp_prediction = 0.01  # 1%
        sl_prediction = 0.005  # 0.5%
    
    # Ensure minimum values
    tp_prediction = max(tp_prediction, 0.003)  # Minimum 0.3%
    sl_prediction = max(sl_prediction, 0.002)  # Minimum 0.2%
    
    # Adjust based on market conditions
    regime = market_conditions['regime']
    
    # TP/SL adjustments for different market regimes
    tp_adjustments = {
        'trending': 1.2,     # Extend targets in trends
        'mean_reverting': 0.8,  # Tighter targets in mean reversion
        'volatile': 1.3,     # Wider targets in volatility
        'quiet': 0.9         # Tighter targets in quiet markets
    }
    
    sl_adjustments = {
        'trending': 0.9,     # Tighter stops in trends
        'mean_reverting': 1.1,  # Wider stops in mean reversion
        'volatile': 1.3,     # Much wider stops in volatility
        'quiet': 0.9         # Tighter stops in quiet markets
    }
    
    # Apply adjustments
    tp_multiplier = tp_adjustments.get(regime, 1.0)
    sl_multiplier = sl_adjustments.get(regime, 1.0)
    
    # Further adjust based on current volatility relative to historical volatility
    volatility_ratio = current_volatility / 0.001  # Compare to baseline of 0.1%
    if volatility_ratio > 1.5:
        # Higher volatility - widen both TP and SL
        tp_multiplier *= 1.1
        sl_multiplier *= 1.2
    elif volatility_ratio < 0.5:
        # Lower volatility - tighten both
        tp_multiplier *= 0.9
        sl_multiplier *= 0.9
    
    # Apply global TP/SL factors from config
    tp_multiplier *= tp_factor
    sl_multiplier *= sl_factor
    
    # Calculate final targets
    if signal_type == 'buy':
        take_profit = price * (1 + tp_prediction * tp_multiplier)
        stop_loss = price * (1 - sl_prediction * sl_multiplier)
    else:  # 'sell'
        take_profit = price * (1 - tp_prediction * tp_multiplier)
        stop_loss = price * (1 + sl_prediction * sl_multiplier)
    
    return take_profit, stop_loss

def update_trailing_stop(current_price, trade_direction, trade_data):
    """Update trailing stop loss based on price movement"""
    initial_stop = trade_data['stop_loss']
    highest_price = trade_data.get('highest_price', current_price)
    lowest_price = trade_data.get('lowest_price', current_price)
    
    # Track highest/lowest price since trade entry
    if trade_direction == 'buy':
        highest_price = max(highest_price, current_price)
        # Calculate trailing stop with the tsl_factor
        trailing_level = highest_price * (1 - tsl_factor)
        # Only move stop loss up, never down
        new_stop = max(initial_stop, trailing_level)
    else:  # 'sell'
        lowest_price = min(lowest_price, current_price)
        # Calculate trailing stop with the tsl_factor
        trailing_level = lowest_price * (1 + tsl_factor)
        # Only move stop loss down, never up
        new_stop = min(initial_stop, trailing_level) if initial_stop > 0 else trailing_level
    
    return new_stop, highest_price, lowest_price

# ========== BACKTESTING ENGINE ==========
def backtest(df):
    """Backtesting engine for trading strategy"""
    print("Starting backtest...")
    
    # Prepare data
    df = compute_indicators(df)
    df = label_data(df)
    df = create_feature_set(df)
    
    # Reset index to avoid issues
    df = df.reset_index()
    
    # Initialize tracking variables
    balance = initial_balance
    trades = []
    active_trade = None
    model_retrain_counter = 0
    performance_metrics = []
    
    # Select features for model
    feature_cols = [
        'rsi', 'macd', 'macd_hist', 'bb_width', 'ema_fast_slope', 
        'price_vs_ema_fast', 'price_vs_ema_slow', 'adx', 'atr_pct',
        'bb_position', 'rsi_slope', 'macd_hist_slope', 'ema_cross',
        'macd_cross', 'rsi_cat', 'trend_strength', 'three_bull_candles',
        'three_bear_candles', 'bull_vol_confirm', 'bear_vol_confirm',
        'adx_directional_trend', 'align_with_higher_tf'
    ]
    
    # Available features check
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Initialize models
    rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler = load_or_initialize_models()
    
    # Initial training window size
    train_size = 500  # Start with first 500 data points
    
    # Process each candle
    for i in range(train_size, len(df)):
        current = df.iloc[i]
        
        # Only process during allowed trading hours
        if not time_filter(current.get('timestamp', current.name)):
            continue
            
        # Retrain models periodically
        if model_retrain_counter >= retrain_every:
            train_end = i
            train_start = max(0, train_end - 2000)  # Use last 2000 candles for training
            train_df = df.iloc[train_start:train_end]
            
            # Prepare training data
            X_train = train_df[feature_cols]
            y_train = train_df['label']
            
            # Get potential gain/loss for TP/SL models
            y_train_tp = train_df['potential_gain']
            y_train_sl = train_df['potential_loss']
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Check if we have both positive and negative samples
            if len(y_train.unique()) > 1:
                # Train models
                rf_clf.fit(X_train_scaled, y_train)
                xgb_clf.fit(X_train_scaled, y_train)
                lgbm_clf.fit(X_train_scaled, y_train)
                tp_model.fit(X_train_scaled, y_train_tp)
                sl_model.fit(X_train_scaled, y_train_sl)
                
                # Log training metrics
                try:
                    y_pred = rf_clf.predict(X_train_scaled)
                    train_auc = roc_auc_score(y_train, rf_clf.predict_proba(X_train_scaled)[:, 1])
                    train_precision = precision_score(y_train, y_pred, zero_division=0)
                    train_recall = recall_score(y_train, y_pred, zero_division=0)
                    train_f1 = f1_score(y_train, y_pred, zero_division=0)
                    
                    print(f"Model retrained at candle {i}. "
                          f"AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, "
                          f"Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
                except Exception as e:
                    print(f"Error calculating training metrics: {e}")
            
            model_retrain_counter = 0
        else:
            model_retrain_counter += 1
        
        # Extract current price data
        current_price = current['close']
        timestamp = current.get('timestamp', current.name)
        
        # Process active trade if exists
        if active_trade:
            # Get the trade info
            trade_direction = active_trade['direction']
            entry_price = active_trade['entry_price']
            stop_loss = active_trade['stop_loss']
            take_profit = active_trade['take_profit']
            
            # Calculate current P&L
            if trade_direction == 'buy':
                current_pnl_pct = (current_price - entry_price) / entry_price
                # Check if stop loss hit
                if current_price <= stop_loss:
                    # Close trade at stop loss
                    pnl_amount = active_trade['position_size'] * (stop_loss - entry_price)
                    balance += active_trade['position_size'] * stop_loss
                    active_trade['exit_price'] = stop_loss
                    active_trade['exit_time'] = timestamp
                    active_trade['pnl'] = pnl_amount
                    active_trade['pnl_pct'] = (stop_loss - entry_price) / entry_price
                    active_trade['exit_reason'] = 'stop_loss'
                    trades.append(active_trade)
                    active_trade = None
                # Check if take profit hit
                elif current_price >= take_profit:
                    # Close trade at take profit
                    pnl_amount = active_trade['position_size'] * (take_profit - entry_price)
                    balance += active_trade['position_size'] * take_profit
                    active_trade['exit_price'] = take_profit
                    active_trade['exit_time'] = timestamp
                    active_trade['pnl'] = pnl_amount
                    active_trade['pnl_pct'] = (take_profit - entry_price) / entry_price
                    active_trade['exit_reason'] = 'take_profit'
                    trades.append(active_trade)
                    active_trade = None
                else:
                    # Update trailing stop if needed
                    new_stop, highest_price, lowest_price = update_trailing_stop(
                        current_price, trade_direction, active_trade
                    )
                    active_trade['stop_loss'] = new_stop
                    active_trade['highest_price'] = highest_price
                    active_trade['lowest_price'] = lowest_price
            
            else:  # sell trade
                current_pnl_pct = (entry_price - current_price) / entry_price
                # Check if stop loss hit
                if current_price >= stop_loss:
                    # Close trade at stop loss
                    pnl_amount = active_trade['position_size'] * (entry_price - stop_loss)
                    balance += active_trade['position_size'] * (2 * entry_price - stop_loss)
                    active_trade['exit_price'] = stop_loss
                    active_trade['exit_time'] = timestamp
                    active_trade['pnl'] = pnl_amount
                    active_trade['pnl_pct'] = (entry_price - stop_loss) / entry_price
                    active_trade['exit_reason'] = 'stop_loss'
                    trades.append(active_trade)
                    active_trade = None
                # Check if take profit hit
                elif current_price <= take_profit:
                    # Close trade at take profit
                    pnl_amount = active_trade['position_size'] * (entry_price - take_profit)
                    balance += active_trade['position_size'] * (2 * entry_price - take_profit)
                    active_trade['exit_price'] = take_profit
                    active_trade['exit_time'] = timestamp
                    active_trade['pnl'] = pnl_amount
                    active_trade['pnl_pct'] = (entry_price - take_profit) / entry_price
                    active_trade['exit_reason'] = 'take_profit'
                    trades.append(active_trade)
                    active_trade = None
                else:
                    # Update trailing stop if needed
                    new_stop, highest_price, lowest_price = update_trailing_stop(
                        current_price, trade_direction, active_trade
                    )
                    active_trade['stop_loss'] = new_stop
                    active_trade['highest_price'] = highest_price
                    active_trade['lowest_price'] = lowest_price
        
        # Check for new trade signals if no active trade
        if not active_trade:
            # Prepare features for prediction
            X = df.iloc[i:i+1][feature_cols]
            X_scaled = scaler.transform(X)
            
            # Analyze market conditions
            market_conditions = analyze_market_conditions(df, i)
            
            # Generate signals
            models_tuple = (rf_clf, xgb_clf, lgbm_clf)
            signal_data = generate_signal(models_tuple, X_scaled, market_conditions)
            
            # Take action based on signal
            if signal_data['signal'] in ['buy', 'sell'] and signal_data['strength'] >= min_signal_strength:
                # Calculate take profit and stop loss levels
                tp_sl_models = (tp_model, sl_model)
                tp_price, sl_price = calculate_take_profit_stop_loss(
                    tp_sl_models, X_scaled, current_price, 
                    signal_data['signal'], market_conditions, 
                    df.iloc[i]['atr_pct']
                )
                
                # Calculate position size
                pos_size = calculate_position_size(
                    balance, risk_per_trade, current_price, sl_price
                )
                
                # Open new trade if we have a valid signal and position size
                if pos_size > 0:
                    cost = pos_size * current_price if signal_data['signal'] == 'buy' else pos_size * current_price * 2
                    
                    # Check if we have enough balance
                    if cost <= balance:
                        # Open trade
                        active_trade = {
                            'entry_time': timestamp,
                            'direction': signal_data['signal'],
                            'entry_price': current_price,
                            'position_size': pos_size,
                            'take_profit': tp_price,
                            'stop_loss': sl_price,
                            'signal_strength': signal_data['strength'],
                            'signal_confidence': signal_data['confidence'],
                            'market_regime': market_conditions['regime'],
                            'balance_before': balance,
                            'highest_price': current_price,
                            'lowest_price': current_price
                        }
                        
                        # Adjust balance
                        if signal_data['signal'] == 'buy':
                            balance -= pos_size * current_price  # Money spent on buying
                        else:  # 'sell' - short selling
                            # For shorts, we lock up the entry value and potential max loss
                            # This is slightly simplified but serves the purpose
                            balance -= pos_size * current_price  # Lock up equivalent value
                
        # Record performance at regular intervals
        if i % 100 == 0:
            performance_metrics.append({
                'candle': i,
                'timestamp': timestamp,
                'balance': balance,
                'trades_completed': len(trades),
                'active_trade': active_trade is not None
            })
    
    # Close any open trade at the end of backtest
    if active_trade:
        last_price = df.iloc[-1]['close']
        if active_trade['direction'] == 'buy':
            pnl_amount = active_trade['position_size'] * (last_price - active_trade['entry_price'])
            balance += active_trade['position_size'] * last_price
        else:  # 'sell'
            pnl_amount = active_trade['position_size'] * (active_trade['entry_price'] - last_price)
            balance += active_trade['position_size'] * (2 * active_trade['entry_price'] - last_price)
        
        active_trade['exit_price'] = last_price
        active_trade['exit_time'] = df.iloc[-1].get('timestamp', df.iloc[-1].name)
        active_trade['pnl'] = pnl_amount
        active_trade['exit_reason'] = 'backtest_end'
        trades.append(active_trade)
    
    # Calculate backtest results
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = sum(t['pnl'] for t in trades)
    profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                     sum(t['pnl'] for t in trades if t['pnl'] < 0)) if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else float('inf')
    
    # Average trade metrics
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
    
    # Calculate maximum drawdown
    equity_curve = [initial_balance]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
    
    max_drawdown_pct = 0
    peak = equity_curve[0]
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
    
    # Print results
    print("\n========== BACKTEST RESULTS ==========")
    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance: ${balance:.2f}")
    print(f"Net Profit: ${balance - initial_balance:.2f} ({((balance/initial_balance)-1)*100:.2f}%)")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average Profit per Trade: ${avg_profit:.2f}")
    print(f"Average Winning Trade: ${avg_win:.2f}")
    print(f"Average Losing Trade: ${avg_loss:.2f}")
    print(f"Maximum Drawdown: {max_drawdown_pct:.2%}")
    
    # Return results and trade data for further analysis
    return {
        'final_balance': balance,
        'net_profit': balance - initial_balance,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_profit': avg_profit,
        'max_drawdown_pct': max_drawdown_pct,
        'trades': trades,
        'equity_curve': equity_curve,
        'performance_metrics': performance_metrics
    }

# ========== VISUALIZATION ==========
def visualize_results(backtest_results, df):
    """Visualize backtest results with matplotlib"""
    # Create a directory for plots
    os.makedirs("analysis/plots", exist_ok=True)
    
    # 1. Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_results['equity_curve'])
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Account Balance')
    plt.grid(True)
    plt.savefig('analysis/plots/equity_curve.png')
    plt.close()
    
    # 2. Trade Outcomes Distribution
    pnl_values = [trade['pnl'] for trade in backtest_results['trades']]
    plt.figure(figsize=(12, 6))
    sns.histplot(pnl_values, kde=True)
    plt.title('Trade Outcomes Distribution')
    plt.xlabel('Profit/Loss')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('analysis/plots/trade_distribution.png')
    plt.close()
    
    # 3. Win Rate by Market Regime
    regimes = [trade['market_regime'] for trade in backtest_results['trades']]
    outcomes = [1 if trade['pnl'] > 0 else 0 for trade in backtest_results['trades']]
    regime_df = pd.DataFrame({'regime': regimes, 'win': outcomes})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='regime', y='win', data=regime_df, estimator=np.mean)
    plt.title('Win Rate by Market Regime')
    plt.xlabel('Market Regime')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('analysis/plots/winrate_by_regime.png')
    plt.close()
    
    # 4. Trade Duration Analysis
    try:
        durations = []
        for trade in backtest_results['trades']:
            if isinstance(trade['entry_time'], (pd.Timestamp, datetime)) and isinstance(trade['exit_time'], (pd.Timestamp, datetime)):
                # 4. Trade Duration Analysis (continued)
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 60  # in minutes
                durations.append(duration)
        
        plt.figure(figsize=(12, 6))
        sns.histplot(durations, kde=True)
        plt.title('Trade Duration Distribution')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('analysis/plots/trade_duration.png')
        plt.close()
    except Exception as e:
        print(f"Error creating duration analysis: {e}")
    
    # 5. Signal Strength vs. Outcome
    strengths = [trade['signal_strength'] for trade in backtest_results['trades']]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=strengths, y=pnl_values)
    plt.title('Signal Strength vs. Trade Outcome')
    plt.xlabel('Signal Strength')
    plt.ylabel('Profit/Loss')
    plt.grid(True)
    plt.savefig('analysis/plots/strength_vs_outcome.png')
    plt.close()
    
    # Save detailed trade statistics to CSV
    trade_df = pd.DataFrame(backtest_results['trades'])
    trade_df.to_csv('analysis/trade_details.csv', index=False)
    
    # Print summary statistics
    print("\n========== ADDITIONAL STATISTICS ==========")
    
    # Average trade duration
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"Average Trade Duration: {avg_duration:.2f} minutes")
    
    # Best and worst trades
    if pnl_values:
        best_trade = max(pnl_values)
        worst_trade = min(pnl_values)
        print(f"Best Trade: ${best_trade:.2f}")
        print(f"Worst Trade: ${worst_trade:.2f}")
    
    # Win rate by direction
    long_trades = [t for t in backtest_results['trades'] if t['direction'] == 'buy']
    short_trades = [t for t in backtest_results['trades'] if t['direction'] == 'sell']
    
    long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
    short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
    
    long_win_rate = long_wins / len(long_trades) if long_trades else 0
    short_win_rate = short_wins / len(short_trades) if short_trades else 0
    
    print(f"Long Trades: {len(long_trades)}, Win Rate: {long_win_rate:.2%}")
    print(f"Short Trades: {len(short_trades)}, Win Rate: {short_win_rate:.2%}")
    
    # Win rate by exit reason
    exit_reasons = {}
    for trade in backtest_results['trades']:
        reason = trade['exit_reason']
        win = trade['pnl'] > 0
        
        if reason not in exit_reasons:
            exit_reasons[reason] = {'total': 0, 'wins': 0}
        
        exit_reasons[reason]['total'] += 1
        if win:
            exit_reasons[reason]['wins'] += 1
    
    print("\nWin Rate by Exit Reason:")
    for reason, stats in exit_reasons.items():
        win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
        print(f"  {reason}: {win_rate:.2%} ({stats['wins']}/{stats['total']})")

# ========== PARAMETER OPTIMIZATION ==========
def objective(trial, data, initial_balance):
    """Objective function for hyperparameter optimization"""
    # Define the hyperparameters to optimize
    params = {
        'confidence_threshold': trial.suggest_float('confidence_threshold', 0.7, 0.95),
        'tp_factor': trial.suggest_float('tp_factor', 1.0, 3.0),
        'sl_factor': trial.suggest_float('sl_factor', 0.5, 1.5),
        'tsl_factor': trial.suggest_float('tsl_factor', 0.1, 0.5),
        'min_signal_strength': trial.suggest_int('min_signal_strength', 30, 70),
        'risk_per_trade': trial.suggest_float('risk_per_trade', 0.005, 0.02)
    }
    
    # Set global parameters for the backtest
    global confidence_threshold, tp_factor, sl_factor, tsl_factor, min_signal_strength, risk_per_trade
    confidence_threshold = params['confidence_threshold']
    tp_factor = params['tp_factor']
    sl_factor = params['sl_factor']
    tsl_factor = params['tsl_factor']
    min_signal_strength = params['min_signal_strength']
    risk_per_trade = params['risk_per_trade']
    
    # Run a backtest with the current parameters
    # We'll use a simplified version to save time
    test_data = data.copy()
    
    # Prepare data
    test_data = compute_indicators(test_data)
    test_data = label_data(test_data)
    test_data = create_feature_set(test_data)
    
    # Reset index
    test_data = test_data.reset_index()
    
    # Train models on a portion of the data
    train_size = int(len(test_data) * 0.5)  # Use half for training
    train_df = test_data.iloc[:train_size]
    test_df = test_data.iloc[train_size:]
    
    # Select features
    feature_cols = [
        'rsi', 'macd', 'macd_hist', 'bb_width', 'ema_fast_slope', 
        'price_vs_ema_fast', 'price_vs_ema_slow', 'adx', 'atr_pct',
        'bb_position', 'rsi_slope', 'macd_hist_slope', 'ema_cross',
        'macd_cross', 'rsi_cat', 'trend_strength'
    ]
    
    # Available features check
    feature_cols = [col for col in feature_cols if col in test_data.columns]
    
    # Initialize models
    rf_clf, xgb_clf, lgbm_clf, tp_model, sl_model, scaler = load_or_initialize_models()
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    
    # Get potential gain/loss for TP/SL models
    y_train_tp = train_df['potential_gain']
    y_train_sl = train_df['potential_loss']
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train models if we have both positive and negative samples
    if len(y_train.unique()) > 1:
        rf_clf.fit(X_train_scaled, y_train)
        xgb_clf.fit(X_train_scaled, y_train)
        lgbm_clf.fit(X_train_scaled, y_train)
        tp_model.fit(X_train_scaled, y_train_tp)
        sl_model.fit(X_train_scaled, y_train_sl)
    
    # Run a simplified backtest on test data
    balance = initial_balance
    trades = []
    active_trade = None
    
    for i in range(len(test_df)):
        current = test_df.iloc[i]
        current_price = current['close']
        
        # Process active trade if exists
        if active_trade:
            # Get the trade info
            trade_direction = active_trade['direction']
            entry_price = active_trade['entry_price']
            stop_loss = active_trade['stop_loss']
            take_profit = active_trade['take_profit']
            
            # Check if stop loss or take profit hit
            if trade_direction == 'buy':
                if current_price <= stop_loss:
                    # Close trade at stop loss
                    pnl_amount = active_trade['position_size'] * (stop_loss - entry_price)
                    balance += active_trade['position_size'] * stop_loss
                    active_trade['pnl'] = pnl_amount
                    trades.append(active_trade)
                    active_trade = None
                elif current_price >= take_profit:
                    # Close trade at take profit
                    pnl_amount = active_trade['position_size'] * (take_profit - entry_price)
                    balance += active_trade['position_size'] * take_profit
                    active_trade['pnl'] = pnl_amount
                    trades.append(active_trade)
                    active_trade = None
                else:
                    # Update trailing stop
                    new_stop, highest_price, lowest_price = update_trailing_stop(
                        current_price, trade_direction, active_trade
                    )
                    active_trade['stop_loss'] = new_stop
                    active_trade['highest_price'] = highest_price
                    active_trade['lowest_price'] = lowest_price
            else:  # sell trade
                if current_price >= stop_loss:
                    # Close trade at stop loss
                    pnl_amount = active_trade['position_size'] * (entry_price - stop_loss)
                    balance += active_trade['position_size'] * (2 * entry_price - stop_loss)
                    active_trade['pnl'] = pnl_amount
                    trades.append(active_trade)
                    active_trade = None
                elif current_price <= take_profit:
                    # Close trade at take profit
                    pnl_amount = active_trade['position_size'] * (entry_price - take_profit)
                    balance += active_trade['position_size'] * (2 * entry_price - take_profit)
                    active_trade['pnl'] = pnl_amount
                    trades.append(active_trade)
                    active_trade = None
                else:
                    # Update trailing stop
                    new_stop, highest_price, lowest_price = update_trailing_stop(
                        current_price, trade_direction, active_trade
                    )
                    active_trade['stop_loss'] = new_stop
                    active_trade['highest_price'] = highest_price
                    active_trade['lowest_price'] = lowest_price
        
        # Check for new signals if no active trade
        if not active_trade:
            # Prepare features
            X = test_df.iloc[i:i+1][feature_cols]
            X_scaled = scaler.transform(X)
            
            # Analyze market conditions
            market_conditions = analyze_market_conditions(test_df, i)
            
            # Generate signal
            models_tuple = (rf_clf, xgb_clf, lgbm_clf)
            signal_data = generate_signal(models_tuple, X_scaled, market_conditions)
            
            # Take action based on signal
            if signal_data['signal'] in ['buy', 'sell'] and signal_data['strength'] >= min_signal_strength:
                # Calculate TP/SL
                tp_sl_models = (tp_model, sl_model)
                tp_price, sl_price = calculate_take_profit_stop_loss(
                    tp_sl_models, X_scaled, current_price, 
                    signal_data['signal'], market_conditions, 
                    test_df.iloc[i]['atr_pct']
                )
                
                # Calculate position size
                pos_size = calculate_position_size(
                    balance, risk_per_trade, current_price, sl_price
                )
                
                # Open new trade if valid
                if pos_size > 0:
                    cost = pos_size * current_price if signal_data['signal'] == 'buy' else pos_size * current_price * 2
                    
                    if cost <= balance:
                        active_trade = {
                            'direction': signal_data['signal'],
                            'entry_price': current_price,
                            'position_size': pos_size,  
                            'take_profit': tp_price,
                            'stop_loss': sl_price,
                            'highest_price': current_price,
                            'lowest_price': current_price
                        }
                        
                        if signal_data['signal'] == 'buy':
                            balance -= pos_size * current_price
                        else:  # 'sell'
                            balance -= pos_size * current_price
    
    # Close any open trade at the end
    if active_trade:
        last_price = test_df.iloc[-1]['close']
        if active_trade['direction'] == 'buy':
            pnl_amount = active_trade['position_size'] * (last_price - active_trade['entry_price'])
            balance += active_trade['position_size'] * last_price
        else:  # 'sell'
            pnl_amount = active_trade['position_size'] * (active_trade['entry_price'] - last_price)
            balance += active_trade['position_size'] * (2 * active_trade['entry_price'] - last_price)
        
        active_trade['pnl'] = pnl_amount
        trades.append(active_trade)
    
    # Calculate metrics for optimization
    total_trades = len(trades)
    if total_trades == 0:
        return -1  # Penalize if no trades are taken
    
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades
    
    # Calculate profit metrics
    total_profit = sum(t['pnl'] for t in trades)
    profit_factor = abs(sum(t['pnl'] for t in trades if t['pnl'] > 0) / 
                     sum(t['pnl'] for t in trades if t['pnl'] < 0)) if sum(t['pnl'] for t in trades if t['pnl'] < 0) != 0 else float('inf')
    
    # Calculate equity curve for drawdown
    equity_curve = [initial_balance]
    for trade in trades:
        equity_curve.append(equity_curve[-1] + trade['pnl'])
    
    max_drawdown_pct = 0
    peak = equity_curve[0]
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)
    
    # Create a combined objective metric - optimize for risk-adjusted returns
    roi = (balance - initial_balance) / initial_balance
    
    # Penalize high drawdowns and reward high win rates
    if max_drawdown_pct > 0.2:  # Cap drawdown at 20%
        score = roi * win_rate / (max_drawdown_pct * 2)
    else:
        score = roi * win_rate / max(0.01, max_drawdown_pct)
    
    # Penalize strategies with very few trades
    if total_trades < 10:
        score *= (total_trades / 10)
    
    # Print current trial results
    print(f"Trial {trial.number}: Score={score:.4f}, ROI={roi:.2%}, Win Rate={win_rate:.2%}, Trades={total_trades}, Drawdown={max_drawdown_pct:.2%}")
    
    return score

def optimize_parameters(data):
    """Optimize strategy parameters using Optuna"""
    print("Starting parameter optimization...")
    
    # Create a study object
    study = optuna.create_study(direction='maximize')
    
    # Run the optimization
    study.optimize(
        partial(objective, data=data, initial_balance=initial_balance),
        n_trials=30  # Adjust based on available time
    )
    
    # Get the best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    print("\n========== OPTIMIZATION RESULTS ==========")
    print(f"Best Score: {best_value:.4f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save best parameters
    with open("analysis/best_params.txt", "w") as f:
        f.write(f"Best Score: {best_value:.4f}\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    return best_params

# ========== MAIN FUNCTION ==========
def main():
    print("Crypto Trading Bot Backtest")
    print("===========================")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {len(df)} candles")
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Check if optimization is needed
    optimize = input("Run parameter optimization? (y/n): ").lower() == 'y'
    
    if optimize:
        # Run optimization
        best_params = optimize_parameters(df)
        
        # Update global parameters
        global confidence_threshold, tp_factor, sl_factor, tsl_factor, min_signal_strength, risk_per_trade
        confidence_threshold = best_params['confidence_threshold']
        tp_factor = best_params['tp_factor']
        sl_factor = best_params['sl_factor']
        tsl_factor = best_params['tsl_factor']
        min_signal_strength = best_params['min_signal_strength']
        risk_per_trade = best_params['risk_per_trade']
    
    # Run backtest
    results = backtest(df)
    
    # Visualize results
    visualize_results(results, df)
    
    print("\nBacktest completed! Results saved to 'analysis' directory.")

if __name__ == "__main__":
    main()