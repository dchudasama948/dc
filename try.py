import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
import talib
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
warnings.filterwarnings('ignore')

class CryptoStrategyRunner:
    def __init__(self):
        print(f"Initializing Crypto Strategy Runner...")
        print(f"Current time (UTC): 2025-04-25 18:08:37")
        print(f"User: dchudasama948")
        
        self.monthly_results = {}
        self.model_metrics = {}
        
    def load_monthly_data(self, data_folder):
        """Load all monthly CSV files from the specified folder"""
        print("\nLoading monthly data files...")
        
        all_data = []
        for month in range(1, 13):
            try:
                file_path = f"{data_folder}/crypto_data_{month:02d}.csv"
                df = pd.read_csv(file_path)
                
                # Rename columns if they're in Yahoo Finance format
                if 'Date' in df.columns:
                    df = df.rename(columns={
                        'Date': 'datetime',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Adj Close': 'adj_close',
                        'Volume': 'volume'
                    })
                
                # Ensure all required columns exist
                required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    raise ValueError(f"Missing required columns in {file_path}")
                
                df['month'] = month
                all_data.append(df)
                print(f"✓ Loaded data for month {month:02d}")
            except Exception as e:
                print(f"Warning: Could not load month {month:02d} - {e}")
                
        if not all_data:
            raise ValueError("No data could be loaded")
            
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Convert datetime column
        combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])
        
        # Sort by datetime
        combined_data = combined_data.sort_values('datetime')
        
        print(f"\nData shape: {combined_data.shape}")
        print(f"Columns: {combined_data.columns.tolist()}")
        return combined_data

    def add_technical_indicators(self, df):
        """Add technical indicators"""
        try:
            df = df.copy()
            
            # Basic price features
            df['hl2'] = (df['high'] + df['low']) / 2
            df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Convert to numpy arrays for talib
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Add indicators
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_21'] = talib.EMA(close, timeperiod=21)
            df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            df['rsi'] = talib.RSI(close, timeperiod=14)
            
            macd, signal, hist = talib.MACD(close)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            print(f"Error in adding technical indicators: {str(e)}")
            return None

    def feature_engineering(self, df):
        """Create advanced features"""
        try:
            if df is None:
                raise ValueError("Input dataframe is None")
                
            df = df.copy()
            
            # Trend features
            df['trend_strength'] = abs(df['ema_9'] - df['ema_50']) / df['atr']
            df['trend_direction'] = np.where(df['ema_9'] > df['ema_50'], 1, -1)
            
            # Price vs EMA features
            df['price_vs_ema_short'] = (df['close'] - df['ema_9']) / df['close']
            df['price_vs_ema_med'] = (df['close'] - df['ema_21']) / df['close']
            df['price_vs_ema_long'] = (df['close'] - df['ema_50']) / df['close']
            
            # Momentum features
            df['rsi_trend'] = df['rsi'] - df['rsi'].shift(1)
            df['macd_trend'] = df['macd_hist'] - df['macd_hist'].shift(1)
            
            # Volatility features
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility_ratio'] = df['atr'] / df['close']
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Entry/Exit zones
            df['entry_zone'] = ((df['rsi'] < 40) & 
                            (df['close'] < df['bb_lower']) & 
                            (df['macd_hist'] > 0)).astype(int)
            
            df['exit_zone'] = ((df['rsi'] > 70) | 
                            (df['close'] > df['bb_upper']) | 
                            (df['macd_hist'] < 0)).astype(int)
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            return None

    def generate_signals(self, df):
        """Generate trading signals"""
        try:
            if df is None:
                raise ValueError("Input dataframe is None")
                
            signals = pd.DataFrame(index=df.index)
            
            # Entry conditions
            long_entry = (
                (df['rsi'] < 40) &
                (df['close'] < df['bb_lower']) &
                (df['macd_hist'] > 0) &
                (df['ema_9'] > df['ema_21']) &
                (df['volume_ratio'] > 1.2)
            )
            
            # Exit conditions
            long_exit = (
                (df['rsi'] > 70) |
                (df['close'] > df['bb_upper']) |
                (df['macd_hist'] < 0) |
                (df['ema_9'] < df['ema_21'])
            )
            
            signals['entry_signal'] = long_entry.astype(int)
            signals['exit_signal'] = long_exit.astype(int)
            
            # Risk management levels
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
            
            signals['tsl_price'] = np.where(
                signals['entry_signal'] == 1,
                df['close'] - (signals['sl_distance'] * 0.8),
                0
            )
            
            return signals
            
        except Exception as e:
            print(f"Error in generating signals: {str(e)}")
            return None

    def prepare_monthly_data(self, df):
        """Prepare data for each month"""
        print("\nPreparing monthly data...")
        
        monthly_data = {}
        for month in df['month'].unique():
            try:
                print(f"\nProcessing month {month:02d}")
                month_df = df[df['month'] == month].copy()
                
                # Add indicators
                month_df = self.add_technical_indicators(month_df)
                if month_df is None:
                    raise ValueError("Failed to add technical indicators")
                    
                # Add features
                month_df = self.feature_engineering(month_df)
                if month_df is None:
                    raise ValueError("Failed to add features")
                    
                # Generate signals
                signals = self.generate_signals(month_df)
                if signals is None:
                    raise ValueError("Failed to generate signals")
                
                monthly_data[month] = {
                    'data': month_df,
                    'signals': signals
                }
                print(f"✓ Successfully prepared data for month {month:02d}")
                
            except Exception as e:
                print(f"Error processing month {month:02d}: {str(e)}")
                continue
                
        if not monthly_data:
            raise ValueError("No monthly data could be prepared")
            
        return monthly_data
    def create_model(self):
        """Create the neural network model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, 11)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')  # [entry, exit, position_size]
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, monthly_data):
        """Train the AI model with visualization"""
        print("\nTraining AI Model...")
        
        # Updated feature columns to match what we actually have
        feature_columns = [
            'rsi', 'macd', 'macd_hist', 'bb_position',
            'trend_strength', 'trend_direction', 'volume_ratio',
            'volatility_ratio', 'price_vs_ema_short',
            'price_vs_ema_med', 'price_vs_ema_long'
        ]
        
        # Prepare training data
        X_train = []
        y_train = []
        
        print("Preparing training data...")
        
        for month, data in monthly_data.items():
            df = data['data']
            signals = data['signals']
            
            # Verify all features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                print(f"Warning: Missing features for month {month}: {missing_features}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
                
            # Create sequences of 30 periods
            for i in range(30, len(df)):
                try:
                    sequence = df[feature_columns].values[i-30:i]
                    
                    # Create target variables
                    entry = signals['entry_signal'].iloc[i]
                    exit = signals['exit_signal'].iloc[i]
                    position_size = 1.0 if entry == 1 else 0.0
                    
                    X_train.append(sequence)
                    y_train.append([entry, exit, position_size])
                    
                except Exception as e:
                    print(f"Error processing sequence at index {i}: {str(e)}")
                    continue
        
        if not X_train:
            raise ValueError("No training data could be prepared")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        # Create and train model
        model = self.create_model()
        
        # Update input shape to match our features
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30, len(feature_columns))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return model, history

    def plot_training_history(self, history):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def run_trading_simulation(self, model, monthly_data):
        """Run trading simulation with the trained model"""
        print("\nRunning Trading Simulation...")
        
        feature_columns = [
            'rsi', 'macd', 'macd_hist', 'bb_position',
            'trend_strength', 'trend_direction', 'volume_ratio',
            'volatility_ratio', 'price_vs_ema_short',
            'price_vs_ema_med', 'price_vs_ema_long'
        ]
        
        all_trades = []
        monthly_performance = {}
        
        for month, data in monthly_data.items():
            print(f"\nSimulating month {month}...")
            df = data['data']
            signals = data['signals']
            
            # Initialize variables for this month
            balance = 10000  # Starting balance for each month
            position = 0
            trades = []
            equity_curve = [balance]
            
            # Run simulation
            for i in range(30, len(df)):
                try:
                    sequence = df[feature_columns].values[i-30:i].reshape(1, 30, len(feature_columns))
                    predictions = model.predict(sequence, verbose=0)[0]
                    
                    current_price = df['close'].iloc[i]
                    
                    if position == 0 and predictions[0] > 0.7:  # Entry signal
                        position = 1
                        entry_price = current_price
                        trade = {
                            'entry_time': df['datetime'].iloc[i],
                            'entry_price': entry_price,
                            'position_size': predictions[2]
                        }
                    
                    elif position == 1:
                        # Check exit conditions
                        if (predictions[1] > 0.7 or  # AI exit signal
                            current_price <= signals['sl_price'].iloc[i] or
                            current_price >= signals['tp_price'].iloc[i]):
                            
                            position = 0
                            exit_price = current_price
                            profit_pct = (exit_price - trade['entry_price']) / trade['entry_price']
                            
                            trade.update({
                                'exit_time': df['datetime'].iloc[i],
                                'exit_price': exit_price,
                                'profit_pct': profit_pct,
                                'profit_amount': balance * profit_pct * trade['position_size']
                            })
                            
                            trades.append(trade)
                            balance += trade['profit_amount']
                    
                    equity_curve.append(balance)
                    
                except Exception as e:
                    print(f"Error processing trade at index {i}: {str(e)}")
                    continue
            
            # Calculate monthly metrics
            monthly_performance[month] = {
                'total_trades': len(trades),
                'winning_trades': len([t for t in trades if t['profit_pct'] > 0]),
                'final_balance': balance,
                'return_pct': (balance - 10000) / 10000 * 100,
                'equity_curve': equity_curve,
                'trades': trades
            }
            
            all_trades.extend(trades)
            print(f"Month {month} completed: {len(trades)} trades executed")
        
        return monthly_performance, all_trades

    def display_performance_metrics(self, monthly_performance, all_trades):
        """Display detailed performance metrics"""
        print("\nPerformance Metrics:")
        print("-" * 50)
        
        # Overall metrics
        total_trades = len(all_trades)
        winning_trades = len([t for t in all_trades if t['profit_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        
        # Monthly breakdown
        print("\nMonthly Performance:")
        print("-" * 50)
        
        for month, perf in monthly_performance.items():
            print(f"\nMonth {month}:")
            print(f"Total Trades: {perf['total_trades']}")
            print(f"Winning Trades: {perf['winning_trades']}")
            print(f"Return: {perf['return_pct']:.2f}%")
        
        # Plot results
        self.plot_performance(monthly_performance, all_trades)

    def plot_performance(self, monthly_performance, all_trades):
        """Create detailed performance plots"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Equity Curves by Month',
                'Trade Profit Distribution',
                'Monthly Returns',
                'Win Rate by Month',
                'Trade Duration Distribution',
                'Cumulative Returns'
            )
        )
        
        # 1. Equity Curves
        for month, perf in monthly_performance.items():
            fig.add_trace(
                go.Scatter(
                    y=perf['equity_curve'],
                    name=f'Month {month}',
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # 2. Trade Profit Distribution
        profits = [t['profit_pct'] for t in all_trades]
        fig.add_trace(
            go.Histogram(x=profits, name='Profit Distribution'),
            row=1, col=2
        )
        
        # 3. Monthly Returns
        months = list(monthly_performance.keys())
        returns = [perf['return_pct'] for perf in monthly_performance.values()]
        fig.add_trace(
            go.Bar(x=months, y=returns, name='Monthly Returns'),
            row=2, col=1
        )
        
        # 4. Win Rate by Month
        win_rates = [
            perf['winning_trades']/perf['total_trades'] if perf['total_trades'] > 0 else 0
            for perf in monthly_performance.values()
        ]
        fig.add_trace(
            go.Bar(x=months, y=win_rates, name='Win Rate'),
            row=2, col=2
        )
        
        # 5. Trade Duration Distribution
        durations = [
            (t['exit_time'] - t['entry_time']).total_seconds()/3600 
            for t in all_trades
        ]
        fig.add_trace(
            go.Histogram(x=durations, name='Trade Duration (hours)'),
            row=3, col=1
        )
        
        # 6. Cumulative Returns
        cumulative_returns = np.cumsum([t['profit_pct'] for t in all_trades])
        fig.add_trace(
            go.Scatter(y=cumulative_returns, name='Cumulative Returns'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, width=1200, showlegend=True)
        fig.show()
def main():
    try:
        # Initialize the runner
        runner = CryptoStrategyRunner()
        
        # Load and prepare data
        print("\nStep 1: Loading data...")
        data = runner.load_monthly_data("crypto_data")
        
        print("\nStep 2: Preparing monthly data...")
        monthly_data = runner.prepare_monthly_data(data)
        
        print("\nStep 3: Training AI model...")
        model, history = runner.train_model(monthly_data)
        
        print("\nStep 4: Running trading simulation...")
        monthly_performance, all_trades = runner.run_trading_simulation(model, monthly_data)
        
        print("\nStep 5: Displaying results...")
        runner.display_performance_metrics(monthly_performance, all_trades)
        
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
