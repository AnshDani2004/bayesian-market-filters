import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# Full-Stack Backtest on Binance BTC/USDT 

This notebook represents the capstone of Tier 2. We will pull live, real-world BTC/USDT 1-minute candlestick data from the Binance REST API, run our entire algorithmic filter stack (Kalman + HMM + Particle), generate sizing and directional signals, and backtest the results with a realistic 0.05% taker fee.
"""

code_imports = """import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from src.data.market import BinanceDataClient, Backtester
from src.filters.kalman import AdaptiveKalmanFilter
from src.filters.hmm import RegimeHMM
from src.filters.particle import StochasticVolParticleFilter
from src.signals.generator import SignalGenerator
from src.execution.engine import ExecutionEngine
"""

code_fetch = """# 1. Fetch Data
print("Fetching 1 year of 1-hour BTC/USDT data from Binance...")
df = BinanceDataClient.fetch_data(symbol="BTCUSDT", interval="1h", max_points=10000)
# Slice to exactly 1 year of hours (8760) if we have more
if len(df) > 8760:
    df = df.iloc[-8760:].copy()
    df.reset_index(drop=True, inplace=True)
prices = df['close'].values
returns = df['close'].pct_change().fillna(0).values

print(f"Fetched {len(df)} candles. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
"""

code_run_filters = """# 2. Walk-Forward Validation & Signal Generation
print("Initializing Filters...")

# Calculate train/test split index (50%)
split_idx = len(prices) // 2
is_returns = returns[:split_idx]

# Train HMM offline strictly on In-Sample (IS) data to calibrate regimes without lookahead
hmm = RegimeHMM(n_states=2, random_seed=42)
print("Training HMM on In-Sample Data...")
hmm.fit_baum_welch(is_returns, max_iter=20, tol=1e-3)
prob_volatile_array = hmm.forward_filter(returns)[:, 1]

# Initialize online filters
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
pf = StochasticVolParticleFilter(n_particles=5000, initial_price=prices[0], Q_h=0.0001)
signal_gen = SignalGenerator(base_spread=0.0005, variance_threshold=500.0)
exec_engine = ExecutionEngine(taker_fee_bps=5.0)

signals_naive = np.zeros(len(prices))
signals_threshold = np.zeros(len(prices))
spreads = np.zeros(len(prices))

print("Running Online Filter Stack...")
for t in range(len(prices)):
    y = prices[t]
    
    # Update filters
    kf.predict()
    kf.update(y)
    pf.step(y)
    
    # Get HMM posterior (1 - prob_volatile = prob_quiet)
    prob_quiet = 1.0 - prob_volatile_array[t]
    
    # Generate signals
    sig_dict = signal_gen.generate_signals(
        mid_price=y,
        kf_mean=kf.x[0],
        kf_std=np.sqrt(kf.P[0, 0]),
        kf_drift=kf.x[1],
        prob_quiet=prob_quiet
    )
    
    raw_pos = np.clip(sig_dict['fair_value_signal'], -1.0, 1.0)
    final_pos = raw_pos * sig_dict['position_scalar']
    
    if np.sign(final_pos) != sig_dict['momentum_signal'] and sig_dict['momentum_signal'] != 0:
        final_pos = 0.0
        
    # Naive signal (continuous bleed)
    signals_naive[t] = final_pos
    
    # Executed signal (Alpha Threshold applied)
    signals_threshold[t] = exec_engine.process_signal(
        target_position=final_pos,
        mid_price=y,
        kf_mean=kf.x[0],
        spread_signal=sig_dict['spread_signal']
    )
    
    spreads[t] = sig_dict['spread_signal']

print("Filtering complete.")
"""

code_backtest = """# 3. Backtest IS vs OOS
print("Running Backtest Engine...")

# Run Naive (Taker)
bt_naive = Backtester(taker_fee_bps=5.0) # 0.05% fee
results_naive = bt_naive.run(df, pd.Series(signals_naive))
results_naive['period'] = np.where(np.arange(len(results_naive)) < split_idx, 'IS', 'OOS')

# Run Thresholded (Maker / Limit Order simulation)
bt_thresh = Backtester(taker_fee_bps=0.0) # 0% fee for passive maker orders
results_thresh = bt_thresh.run(df, pd.Series(signals_threshold))
results_thresh['period'] = np.where(np.arange(len(results_thresh)) < split_idx, 'IS', 'OOS')

def print_metrics(res_df, period, title):
    sub_df = res_df[res_df['period'] == period].copy()
    if len(sub_df) == 0: return
    metrics = Backtester.calculate_metrics(sub_df, periods_per_year=8760)
    print(f"\\n=== Backtest Results: {title} ({period}) [N={len(sub_df)}] ===")
    print(f"Sharpe Ratio:  {metrics['sharpe_ratio']:.2f}")
    if 'max_drawdown' in metrics:
        print(f"Max Drawdown:  {metrics['max_drawdown']*100:.2f}%")
        print(f"Hit Rate:      {metrics['hit_rate']*100:.2f}%")
    print(f"Total Net PnL: {metrics['total_net_pnl']*100:.2f}%")

print_metrics(results_naive, 'IS', 'Naive Continuous Bleed')
print_metrics(results_naive, 'OOS', 'Naive Continuous Bleed')
print_metrics(results_thresh, 'IS', 'Alpha Threshold (Maker)')
print_metrics(results_thresh, 'OOS', 'Alpha Threshold (Maker)')
"""

code_plot = """# 4. Plot Cumulative PnL with IS/OOS split
plt.figure(figsize=(14, 7))

# Plot IS/OOS shading
plt.axvspan(results_thresh['timestamp'].iloc[0], results_thresh['timestamp'].iloc[split_idx], 
            color='yellow', alpha=0.1, label='In-Sample (Training)')
plt.axvspan(results_thresh['timestamp'].iloc[split_idx], results_thresh['timestamp'].iloc[-1], 
            color='blue', alpha=0.1, label='Out-of-Sample (Validation)')

# Naive Buy and Hold
buy_hold = (1 + results_naive['asset_return'].fillna(0)).cumprod() - 1
plt.plot(results_naive['timestamp'], buy_hold * 100, label='Buy & Hold BTC', color='gray', alpha=0.6)

# Strategy Naive (Bleed)
plt.plot(results_naive['timestamp'], results_naive['cum_net'] * 100, label='Strategy (Naive: Continuous Bleed)', color='red', alpha=0.7)

# Strategy Threshold (Step-Function)
plt.plot(results_thresh['timestamp'], results_thresh['cum_net'] * 100, label='Strategy (Alpha Thresholded)', color='green', linewidth=2)

plt.title("Execution Engine Impact & Walk-Forward Validation")
plt.xlabel("Date")
plt.ylabel("Cumulative Net Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('backtest_results.png')
# plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_fetch),
    nbf.v4.new_code_cell(code_run_filters),
    nbf.v4.new_code_cell(code_backtest),
    nbf.v4.new_code_cell(code_plot)
]

with open('/Users/ansh/Proj_Res_3/notebook_03_backtest.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 03 created.")
