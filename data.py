import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_crypto_data(start_date, days, base_price=30000):
    """Generate realistic-looking cryptocurrency price data"""
    periods = days * 24  # Hourly data
    dates = pd.date_range(start=start_date, periods=periods, freq='H')
    
    # Generate prices using random walk with drift
    np.random.seed(42)
    returns = np.random.normal(loc=0.0001, scale=0.002, size=periods)
    price_factors = (1 + returns).cumprod()
    closes = base_price * price_factors
    
    # Generate other price components
    df = pd.DataFrame({
        'datetime': dates,
        'open': closes * (1 + np.random.normal(0, 0.001, periods)),
        'high': closes * (1 + abs(np.random.normal(0, 0.002, periods))),
        'low': closes * (1 - abs(np.random.normal(0, 0.002, periods))),
        'close': closes,
        'volume': np.random.normal(1000, 200, periods) * closes
    })
    
    # Ensure price relationships are valid
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def create_monthly_data(year=2025):
    """Create monthly data files"""
    data_folder = "crypto_data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    
    print(f"Generating monthly data for year {year}...")
    
    for month in range(1, 13):
        try:
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            days = (end_date - start_date).days
            
            # Generate data
            df = generate_realistic_crypto_data(start_date, days)
            
            # Save to CSV
            filename = f"{data_folder}/crypto_data_{month:02d}.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Created {filename}")
            
            # Verify file
            test_df = pd.read_csv(filename)
            print(f"  Rows: {len(test_df)}, Columns: {test_df.columns.tolist()}")
            
        except Exception as e:
            print(f"Error generating data for month {month}: {str(e)}")

if __name__ == "__main__":
    current_year = 2025
    create_monthly_data(current_year)
    print("\nSample data creation completed!")
    
    # Verify data structure
    sample_file = "crypto_data/crypto_data_01.csv"
    if os.path.exists(sample_file):
        df = pd.read_csv(sample_file)
        print("\nSample data structure:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
