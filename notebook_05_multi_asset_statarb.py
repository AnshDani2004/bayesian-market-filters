import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.data.market import BinanceDataClient, Backtester
from src.filters.kalman import PairsKalmanFilter


# 1. Fetch Data for Multiple Assets
import os
data_file = 'data/btc_eth_1h_2025_2026.csv'
if os.path.exists(data_file):
    print("Loading data from frozen CSV...")
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
else:
    print("Fetching 1 year of 1-hour BTC and ETH data...")
    df_btc = BinanceDataClient.fetch_data(symbol="BTCUSDT", interval="1h", max_points=10000)
    df_eth = BinanceDataClient.fetch_data(symbol="ETHUSDT", interval="1h", max_points=10000)
    
    df_btc = df_btc.rename(columns={'close': 'btc_close'})[['timestamp', 'btc_close']]
    df_eth = df_eth.rename(columns={'close': 'eth_close'})[['timestamp', 'eth_close']]
    
    # Merge and align timestamps
    df = pd.merge(df_btc, df_eth, on='timestamp', how='inner')
    if len(df) > 8760:
        df = df.iloc[-8760:].copy()
    df.reset_index(drop=True, inplace=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_file, index=False)
    print(f"Saved to {data_file}")

# Calculate log prices
df['log_btc'] = np.log(df['btc_close'])
df['log_eth'] = np.log(df['eth_close'])

prices_btc = df['log_btc'].values
prices_eth = df['log_eth'].values

print(f"Loaded {len(df)} candles. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# 2. Cointegration Filtering & Dynamic Kelly Sizing
print("Running Pairs Kalman Filter...")

kf = PairsKalmanFilter(delta=1e-4, R=1e-3)

target_positions = np.zeros(len(df))
spreads = np.zeros(len(df))
betas = np.zeros(len(df))

for t in range(len(df)):
    x_price = prices_eth[t]
    y_price = prices_btc[t]
    
    # 1. Update Pairs Kalman Filter
    fair_value_y = kf.step(x_price, y_price)
    
    # Extract volatility from Kalman innovation covariance S
    volatility = np.sqrt(kf.S[0, 0])
    
    # Extract current beta (hedge ratio)
    betas[t] = kf.x[1]
    
    # 2. Calculate Spread Z-Score
    spread_error = fair_value_y - y_price
    spreads[t] = spread_error
    
    # Normalize by stochastic volatility (from Kalman S)
    std_safe = max(volatility, 1e-6)
    z_score = spread_error / std_safe
    
    # 3. Dynamic Kelly Sizing
    # Proportional to edge (z-score), bounded between -1.0 and 1.0
    kelly_fraction = np.clip(z_score / 3.0, -1.0, 1.0) 
    
    # Ignore initial warmup period
    if t < 100:
        kelly_fraction = 0.0
        
    target_positions[t] = kelly_fraction

df['target_position'] = target_positions
df['kf_beta'] = betas
df['spread_error'] = spreads
print("Filtering complete.")


# 3. Backtest StatArb Strategy
print("Running Backtest Engine...")

# For Pairs Trading, our return is driven by the Spread Return.
# Since we are modeling log(BTC) = alpha + beta * log(ETH)
# We go long BTC (target_position) and Short ETH (target_position * beta)
# Net Return = target_position * (Return_BTC - beta * Return_ETH)

df['return_btc'] = df['btc_close'].pct_change().fillna(0)
df['return_eth'] = df['eth_close'].pct_change().fillna(0)

# Calculate Spread Return
df['spread_return'] = df['return_btc'] - df['kf_beta'].shift(1).fillna(1.0) * df['return_eth']

# Current Position is Executed Target Position from previous step
df['current_position'] = df['target_position'].shift(1).fillna(0)

# Gross Return
df['gross_return'] = df['current_position'] * df['spread_return']

# Fee Drag: 1.5% maker fee applied to both legs of the pair whenever position size changes
# (Assuming we run passive cross-exchange market making or limit orders)
fee_bps = 1.5 / 10000.0
df['position_change'] = df['current_position'].diff().fillna(0).abs()
# We pay fees on the BTC leg + the ETH leg (approx 2x total notional)
df['fee_drag'] = df['position_change'] * 2.0 * fee_bps

df['net_return'] = df['gross_return'] - df['fee_drag']

df['cum_gross'] = (1 + df['gross_return']).cumprod() - 1
df['cum_net'] = (1 + df['net_return']).cumprod() - 1

# Split IS/OOS
split_idx = len(df) // 2
df['period'] = np.where(np.arange(len(df)) < split_idx, 'IS', 'OOS')

def print_metrics(res_df, period, title):
    sub_df = res_df[res_df['period'] == period].copy()
    if len(sub_df) == 0: return
    net_returns = sub_df['net_return']
    mean_return = net_returns.mean()
    std_return = net_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(8760) if std_return > 0 else 0
    
    cum_net = (1 + net_returns).cumprod()
    max_dd = ((cum_net - cum_net.cummax()) / cum_net.cummax()).min()
    total_pnl = cum_net.iloc[-1] - 1
    
    print(f"\n=== Backtest Results: {title} ({period}) [N={len(sub_df)}] ===")
    print(f"Sharpe Ratio:  {sharpe:.2f}")
    print(f"Max Drawdown:  {max_dd*100:.2f}%")
    print(f"Total Net PnL: {total_pnl*100:.2f}%")

print_metrics(df, 'IS', 'Pairs Trading (Kelly Sizing)')
print_metrics(df, 'OOS', 'Pairs Trading (Kelly Sizing)')


# 4. Plot Cumulative PnL with IS/OOS split
plt.figure(figsize=(14, 7))

plt.axvspan(df['timestamp'].iloc[0], df['timestamp'].iloc[split_idx], 
            color='yellow', alpha=0.1, label='In-Sample (Warmup)')
plt.axvspan(df['timestamp'].iloc[split_idx], df['timestamp'].iloc[-1], 
            color='blue', alpha=0.1, label='Out-of-Sample (Validation)')

plt.plot(df['timestamp'], df['cum_net'] * 100, label='StatArb (Kelly Sizing)', color='green', linewidth=2)

plt.title("Multi-Asset Statistical Arbitrage (BTC/ETH)")
plt.xlabel("Date")
plt.ylabel("Cumulative Net Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('statarb_results.png')
# plt.show()

