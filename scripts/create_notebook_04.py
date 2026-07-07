import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# Machine Learning (Gradient Boosting) Overlay

In this notebook, we extend our Bayesian filtering framework by implementing a supervised Machine Learning overlay. 
Rather than using rigid heuristic thresholds for position sizing, we train a `GradientBoostingClassifier` to ingest the continuous states of our filters (Kalman Z-Score, Volatility, Regime Probability, Macro Trend) and output a dynamic confidence probability of a positive forward return.

We strictly train the ML model on the In-Sample dataset to prevent lookahead bias, and then evaluate its predictive power during the Out-Of-Sample period.
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
from src.signals.ml_generator import MLSignalGenerator
from src.execution.engine import ExecutionEngine
"""

code_fetch = """# 1. Fetch Data
import os
data_file = 'data/btc_1h_2025_2026.csv'
if os.path.exists(data_file):
    print("Loading data from frozen CSV...")
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
else:
    print("Fetching 1 year of 1-hour BTC/USDT data from Binance...")
    df = BinanceDataClient.fetch_data(symbol="BTCUSDT", interval="1h", max_points=10000)
    if len(df) > 8760:
        df = df.iloc[-8760:].copy()
        df.reset_index(drop=True, inplace=True)
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_file, index=False)
    print(f"Saved to {data_file}")

prices = df['close'].values
returns = df['close'].pct_change().fillna(0).values

print(f"Loaded {len(df)} candles. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
"""

code_run_filters = """# 2. Walk-Forward Feature Engineering & Training
print("Initializing Filters...")

split_idx = len(prices) // 2
is_returns = returns[:split_idx]

# Train HMM offline strictly on IS data
hmm = RegimeHMM(n_states=2, random_seed=42)
print("Training HMM on In-Sample Data...")
hmm.fit_baum_welch(is_returns, max_iter=20, tol=1e-3)
prob_volatile_array = hmm.forward_filter(returns)[:, 1]

# Initialize Kalman
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)

# Compute Macro Trend (50-SMA)
sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0]).values
macro_trend_array = np.where(prices > sma_50, 1, -1)
macro_trend_array[:50] = 0

print("Collecting features for ML...")
X = []
y = []

# Collect features up to t-1
for t in range(len(prices) - 1):
    curr_price = prices[t]
    kf.predict()
    kf.update(curr_price)
    
    kf_mean = kf.x[0]
    kf_std = np.sqrt(kf.P[0,0])
    prob_quiet = 1.0 - prob_volatile_array[t]
    macro_trend = macro_trend_array[t]
    fv_zscore = (kf_mean - curr_price) / max(kf_std, 1e-6)
    
    feature_vector = [fv_zscore, kf_std / curr_price, prob_quiet, macro_trend]
    
    forward_return = prices[t+1] / curr_price - 1.0
    label = 1 if forward_return > 0 else 0
    
    X.append(feature_vector)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Training ML Signal Generator on IS data...")
ml_gen = MLSignalGenerator(confidence_threshold=0.55)
# Train strictly on IS data (up to split_idx - 1)
X_train = X[:split_idx-1]
y_train = y[:split_idx-1]
ml_gen.train(X_train, y_train)

print("Generating Online Signals...")
exec_engine = ExecutionEngine(taker_fee_bps=5.0)
signals_ml = np.zeros(len(prices))

# Fast-forward filter states to generate proper signals online
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
for t in range(len(prices)):
    curr_price = prices[t]
    kf.predict()
    kf.update(curr_price)
    
    kf_mean = kf.x[0]
    kf_std = np.sqrt(kf.P[0,0])
    prob_quiet = 1.0 - prob_volatile_array[t]
    macro_trend = macro_trend_array[t]
    fv_zscore = (kf_mean - curr_price) / max(kf_std, 1e-6)
    
    features = np.array([fv_zscore, kf_std / curr_price, prob_quiet, macro_trend])
    
    sig_dict = ml_gen.generate_signals(features, base_spread=0.0005)
    
    signals_ml[t] = exec_engine.process_signal(
        target_position=sig_dict['target_position'],
        mid_price=curr_price,
        kf_mean=kf_mean,
        spread_signal=sig_dict['spread_signal']
    )

print("ML Filtering complete.")
"""

code_backtest = """# 3. Backtest IS vs OOS
print("Running Backtest Engine...")

# Run backtest with 1.5 bps fee
bt_ml = Backtester(taker_fee_bps=1.5) 
results_ml = bt_ml.run(df, pd.Series(signals_ml))
results_ml['period'] = np.where(np.arange(len(results_ml)) < split_idx, 'IS', 'OOS')

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

print_metrics(results_ml, 'IS', 'ML Overlay (Maker)')
print_metrics(results_ml, 'OOS', 'ML Overlay (Maker)')
"""

code_plot = """# 4. Plot Cumulative PnL with IS/OOS split
plt.figure(figsize=(14, 7))

# Plot IS/OOS shading
plt.axvspan(results_ml['timestamp'].iloc[0], results_ml['timestamp'].iloc[split_idx], 
            color='yellow', alpha=0.1, label='In-Sample (Training)')
plt.axvspan(results_ml['timestamp'].iloc[split_idx], results_ml['timestamp'].iloc[-1], 
            color='blue', alpha=0.1, label='Out-of-Sample (Validation)')

# Buy and Hold
buy_hold = (1 + results_ml['asset_return'].fillna(0)).cumprod() - 1
plt.plot(results_ml['timestamp'], buy_hold * 100, label='Buy & Hold BTC', color='gray', alpha=0.6)

# Strategy ML
plt.plot(results_ml['timestamp'], results_ml['cum_net'] * 100, label='Strategy (ML Overlay)', color='purple', linewidth=2)

plt.title("Machine Learning Overlay & Walk-Forward Validation")
plt.xlabel("Date")
plt.ylabel("Cumulative Net Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_backtest_results.png')
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

with open('/Users/ansh/Proj_Res_3/notebook_04_ml_overlay.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook 04 created.")
