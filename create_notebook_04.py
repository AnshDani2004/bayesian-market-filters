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

from market import BinanceDataClient, Backtester
from kalman import AdaptiveKalmanFilter
from hmm import RegimeHMM
from particle import StochasticVolParticleFilter
from signals.generator import SignalGenerator
from execution import ExecutionEngine
"""

code_fetch = """# 1. Fetch Data
print("Fetching 10,000 minutes of BTC/USDT data from Binance...")
df = BinanceDataClient.fetch_data(symbol="BTCUSDT", interval="1m", max_points=10000)
prices = df['close'].values
returns = df['close'].pct_change().fillna(0).values

print(f"Fetched {len(df)} candles. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
"""

code_run_filters = """# 2. Run Filters & Signal Generator
print("Initializing Filters...")

# Train HMM offline on the dataset to calibrate regimes
hmm = RegimeHMM(n_states=2, random_seed=42)
print("Training HMM...")
hmm.fit_baum_welch(returns, max_iter=20, tol=1e-3)
prob_volatile_array = hmm.forward_filter(returns)[:, 1]

# Initialize online filters
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
pf = StochasticVolParticleFilter(n_particles=1000, initial_price=prices[0], Q_h=0.0001, R=100.0)
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

code_backtest = """# 3. Backtest
print("Running Backtest Engine...")
# Run Naive (Taker)
bt_naive = Backtester(taker_fee_bps=5.0) # 0.05% fee
results_naive = bt_naive.run(df, pd.Series(signals_naive))
metrics_naive = Backtester.calculate_metrics(results_naive)

# Run Thresholded (Maker / Limit Order simulation)
bt_thresh = Backtester(taker_fee_bps=0.0) # 0% fee for passive maker orders
results_thresh = bt_thresh.run(df, pd.Series(signals_threshold))
metrics_thresh = Backtester.calculate_metrics(results_thresh)

print("\\n=== Backtest Results: Naive (Continuous Bleed) ===")
print(f"Sharpe Ratio:  {metrics_naive['sharpe_ratio']:.2f}")
print(f"Total Net PnL: {metrics_naive['total_net_pnl']*100:.2f}%")

print("\\n=== Backtest Results: Alpha Threshold (Execution Engine) ===")
print(f"Sharpe Ratio:  {metrics_thresh['sharpe_ratio']:.2f}")
print(f"Max Drawdown:  {metrics_thresh['max_drawdown']*100:.2f}%")
print(f"Hit Rate:      {metrics_thresh['hit_rate']*100:.2f}%")
print(f"Total Net PnL: {metrics_thresh['total_net_pnl']*100:.2f}%")

# Regime Conditional PnL for Threshold
results_thresh['regime'] = np.where(prob_volatile_array > 0.5, 'Volatile', 'Quiet')
regime_pnl = results_thresh.groupby('regime')['net_return'].sum() * 100
print("\\n=== Regime-Conditional Net PnL (%) [Thresholded] ===")
print(regime_pnl)
"""

code_plot = """# 4. Plot Cumulative PnL
plt.figure(figsize=(12, 6))

# Naive Buy and Hold
buy_hold = (1 + results_naive['asset_return'].fillna(0)).cumprod() - 1
plt.plot(results_naive['timestamp'], buy_hold * 100, label='Buy & Hold BTC', color='gray', alpha=0.6)

# Strategy Naive (Bleed)
plt.plot(results_naive['timestamp'], results_naive['cum_net'] * 100, label='Strategy (Naive: Continuous Bleed)', color='red', alpha=0.7)

# Strategy Threshold (Step-Function)
plt.plot(results_thresh['timestamp'], results_thresh['cum_net'] * 100, label='Strategy (Alpha Thresholded)', color='green', linewidth=2)

plt.title("Execution Engine Impact: Eliminating Fee Drag via Alpha Threshold")
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

with open('/Users/ansh/Proj_Res_3/notebook_04_backtest.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 04 created.")
