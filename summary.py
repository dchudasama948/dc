import pandas as pd
import matplotlib.pyplot as plt

def analyze_trades(file_path, qty=100):
    # Read file (CSV or Excel)
    df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)

    # === CASE 1: Simple 1-column PnL input ===
    if df.shape[1] == 1:
        pnl_col = df.columns[0]
        df = pd.DataFrame({
            'pnl': df[pnl_col] * qty,
            'exit_time': pd.date_range(start='2023-01-01', periods=len(df), freq='D')
        })

    # === CASE 2: Full log with qty, direction, etc ===
    else:
        df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        df = df.dropna(subset=['exit_time', 'pnl'])  # clean up

        # Don't multiply, assume pnl already includes qty

    # === Metrics ===
    total_trades = len(df)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    net_profit = df['pnl'].sum()
    gross_profit = wins['pnl'].sum()
    gross_loss = losses['pnl'].sum()
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    avg_trade = df['pnl'].mean()
    avg_win = wins['pnl'].mean() if not wins.empty else 0
    avg_loss = losses['pnl'].mean() if not losses.empty else 0
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()

    # Equity & Drawdown
    df['equity'] = df['pnl'].cumsum()
    df['roll_max'] = df['equity'].cummax()
    df['drawdown'] = df['equity'] - df['roll_max']
    max_drawdown = df['drawdown'].min()

    # === Summary Output ===
    print("========== Strategy Tester Summary ==========")
    print(f"Total Trades:       {total_trades}")
    print(f"Win Rate:           {win_rate:.2f}%")
    print(f"Net Profit:         {net_profit:.2f}")
    print(f"Gross Profit:       {gross_profit:.2f}")
    print(f"Gross Loss:         {gross_loss:.2f}")
    print(f"Profit Factor:      {profit_factor:.2f}")
    print(f"Avg Trade:          {avg_trade:.2f}")
    print(f"Avg Win:            {avg_win:.2f}")
    print(f"Avg Loss:           {avg_loss:.2f}")
    print(f"Best Trade:         {best_trade:.2f}")
    print(f"Worst Trade:        {worst_trade:.2f}")
    print(f"Max Drawdown:       {max_drawdown:.2f}")

    # === Plot Equity Curve ===
    plt.figure(figsize=(12, 5))
    plt.plot(df['exit_time'], df['equity'], label='Equity Curve', color='blue')
    plt.fill_between(df['exit_time'], df['equity'], df['roll_max'], color='red', alpha=0.3, label='Drawdown')
    plt.title("Equity Curve")
    plt.xlabel("Exit Time")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Entry ===
if __name__ == "__main__":
    analyze_trades('trade_log.csv', qty=100)