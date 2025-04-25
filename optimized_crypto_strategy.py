import pandas as pd
import numpy as np
import talib
from datetime import datetime
from advanced_crypto_trading_bot import AdvancedTradingModel  # Using the previous bot as base

class EnhancedTradingStrategy(AdvancedTradingModel):
    def __init__(self):
        super().__init__()
        self.strategy_params = {
            'rsi_period': 14,
            'ema_short': 9,
            'ema_medium': 21,
            'ema_long': 50,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14
        }

    def add_technical_indicators(self, df):
        """
        Add technical indicators for the strategy
        """
        try:
            # Price action features
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Trend Indicators
            df['ema_short'] = talib.EMA(df['close'], timeperiod=self.strategy_params['ema_short'])
            df['ema_medium'] = talib.EMA(df['close'], timeperiod=self.strategy_params['ema_medium'])
            df['ema_long'] = talib.EMA(df['close'], timeperiod=self.strategy_params['ema_long'])
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'],
                fastperiod=self.strategy_params['macd_fast'],
                slowperiod=self.strategy_params['macd_slow'],
                signalperiod=self.strategy_params['macd_signal']
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.strategy_params['rsi_period'])
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                df['close'],
                timeperiod=self.strategy_params['bb_period'],
                nbdevup=self.strategy_params['bb_std'],
                nbdevdn=self.strategy_params['bb_std']
            )
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # ATR for volatility
            df['atr'] = talib.ATR(
                df['high'],
                df['low'],
                df['close'],
                timeperiod=self.strategy_params['atr_period']
            )
            
            return df
            
        except Exception as e:
            print(f"Error in adding technical indicators: {e}")
            return None

    def feature_engineering(self, df):
        """
        Create advanced features for the model
        """
        try:
            # Trend Features
            df['trend_strength'] = abs(df['ema_short'] - df['ema_long']) / df['atr']
            df['trend_direction'] = np.where(df['ema_short'] > df['ema_long'], 1, -1)
            
            # Price Position Features
            df['price_vs_ema_short'] = (df['close'] - df['ema_short']) / df['atr']
            df['price_vs_ema_medium'] = (df['close'] - df['ema_medium']) / df['atr']
            df['price_vs_ema_long'] = (df['close'] - df['ema_long']) / df['atr']
            
            # Momentum Features
            df['rsi_trend'] = df['rsi'] - df['rsi'].shift(1)
            df['macd_trend'] = df['macd_hist'] - df['macd_hist'].shift(1)
            
            # Volatility Features
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility_ratio'] = df['atr'] / df['close']
            
            # Volume Features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Additional Strategy-Specific Features
            df['entry_zone'] = ((df['rsi'] < 30) & (df['close'] < df['bb_lower'])).astype(int)
            df['exit_zone'] = ((df['rsi'] > 70) & (df['close'] > df['bb_upper'])).astype(int)
            
            # Clean up NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            return None

    def generate_signals(self, df):
        """
        Generate trading signals based on indicators and features
        """
        try:
            signals = pd.DataFrame(index=df.index)
            
            # Entry Conditions
            long_entry_conditions = (
                (df['rsi'] < 40) &  # RSI oversold
                (df['close'] < df['bb_lower']) &  # Price below lower BB
                (df['macd_hist'] > 0) &  # MACD histogram positive
                (df['ema_short'] > df['ema_medium']) &  # Short-term trend up
                (df['volume_ratio'] > 1.2)  # Above average volume
            )
            
            # Exit Conditions
            long_exit_conditions = (
                (df['rsi'] > 70) |  # RSI overbought
                (df['close'] > df['bb_upper']) |  # Price above upper BB
                (df['macd_hist'] < 0) |  # MACD histogram negative
                (df['ema_short'] < df['ema_medium'])  # Short-term trend down
            )
            
            # Generate Signals
            signals['entry_signal'] = long_entry_conditions.astype(int)
            signals['exit_signal'] = long_exit_conditions.astype(int)
            
            # Calculate Stop Loss and Take Profit Levels
            signals['atr_multiple'] = 2.0
            signals['sl_distance'] = df['atr'] * signals['atr_multiple']
            signals['tp_distance'] = df['atr'] * signals['atr_multiple'] * 1.5
            
            signals['sl_price'] = np.where(
                signals['entry_signal'] == 1,
                df['close'] - signals['sl_distance'],
                0
            )
            
            signals['tp_price'] = np.where(
                signals['entry_signal'] == 1,
                df['close'] + signals['tp_distance'],
                0
            )
            
            # Trailing Stop Loss
            signals['tsl_price'] = np.where(
                signals['entry_signal'] == 1,
                df['close'] - (signals['sl_distance'] * 0.8),
                0
            )
            
            return signals
            
        except Exception as e:
            print(f"Error in generating signals: {e}")
            return None

def main():
    print(f"Initializing Enhanced Trading Strategy...")
    print(f"Current time (UTC): {datetime.utcnow()}")
    print(f"User: dchudasama948")
    
    # Initialize the strategy
    strategy = EnhancedTradingStrategy()
    
    # Example usage
    print("\nStrategy Components:")
    print("1. Technical Indicators:")
    print("   - EMA (9, 21, 50)")
    print("   - MACD (12, 26, 9)")
    print("   - RSI (14)")
    print("   - Bollinger Bands (20, 2)")
    print("   - ATR (14)")
    
    print("\n2. Entry Conditions:")
    print("   - RSI < 40")
    print("   - Price below lower Bollinger Band")
    print("   - MACD histogram positive")
    print("   - Short EMA > Medium EMA")
    print("   - Above average volume")
    
    print("\n3. Exit Conditions:")
    print("   - RSI > 70")
    print("   - Price above upper Bollinger Band")
    print("   - MACD histogram negative")
    print("   - Short EMA < Medium EMA")
    
    print("\n4. Risk Management:")
    print("   - Dynamic Stop Loss: 2 * ATR")
    print("   - Take Profit: 3 * ATR")
    print("   - Trailing Stop: 1.6 * ATR")
    
    print("\nTo use this strategy:")
    print("1. Load your cryptocurrency data")
    print("2. Run technical indicators: strategy.add_technical_indicators(df)")
    print("3. Generate features: strategy.feature_engineering(df)")
    print("4. Get signals: strategy.generate_signals(df)")
    print("5. Train the AI model with these signals")
    
    print("\nExample code:")
    print("""
    # Load your data
    df = pd.read_csv('your_crypto_data.csv')
    
    # Initialize strategy
    strategy = EnhancedTradingStrategy()
    
    # Add indicators and features
    df = strategy.add_technical_indicators(df)
    df = strategy.feature_engineering(df)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    # Train the model
    strategy.train_models(X_train, y_train)
    
    # Make predictions
    predictions = strategy.predict_ensemble(new_data)
    """)

if __name__ == "__main__":
    main()