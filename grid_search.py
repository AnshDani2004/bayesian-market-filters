import numpy as np
import pandas as pd
from itertools import product
from src.data.market import BinanceDataClient, Backtester
from src.filters.kalman import AdaptiveKalmanFilter
from src.filters.hmm import RegimeHMM
from src.filters.particle import StochasticVolParticleFilter
from src.signals.generator import SignalGenerator
from src.execution.engine import ExecutionEngine
import warnings
warnings.filterwarnings('ignore')

df = BinanceDataClient.fetch_data(symbol="BTCUSDT", interval="1h", max_points=10000)
if len(df) > 8760: df = df.iloc[-8760:].copy()
df.reset_index(drop=True, inplace=True)
prices = df['close'].values
returns = df['close'].pct_change().fillna(0).values

split_idx = len(prices) // 2
is_prices = prices[:split_idx]
is_returns = returns[:split_idx]

hmm = RegimeHMM(n_states=2, random_seed=42)
hmm.fit_baum_welch(is_returns, max_iter=20, tol=1e-3)
prob_volatile_array = hmm.forward_filter(is_returns)[:, 1]

alphas = [0.001, 0.01, 0.05]
q_hs = [0.0001, 0.001, 0.01]
var_threshs = [1000.0, 5000.0, 10000.0]
spreads = [0.002, 0.005, 0.01]

best_sharpe = -999
best_params = None

print("Starting Grid Search on IS...")
for alpha, q_h, var_thresh, spread in product(alphas, q_hs, var_threshs, spreads):
    kf = AdaptiveKalmanFilter(dt=1.0, initial_price=is_prices[0], adaptive=True, alpha=alpha)
    pf = StochasticVolParticleFilter(n_particles=1000, initial_price=is_prices[0], Q_h=q_h) # 1000 for speed
    signal_gen = SignalGenerator(base_spread=spread, variance_threshold=var_thresh)
    exec_engine = ExecutionEngine(taker_fee_bps=5.0)
    
    signals = np.zeros(len(is_prices))
    for t in range(len(is_prices)):
        y = is_prices[t]
        kf.predict()
        kf.update(y)
        # pf.step(y) # Particle filter is slow and actually not used by SignalGenerator except via Q_h indirectly? Wait, SignalGenerator doesn't use pf!
        prob_quiet = 1.0 - prob_volatile_array[t]
        sig_dict = signal_gen.generate_signals(y, kf.x[0], np.sqrt(kf.P[0,0]), kf.x[1], prob_quiet)
        
        raw_pos = np.clip(sig_dict['fair_value_signal'], -1.0, 1.0)
        final_pos = raw_pos * sig_dict['position_scalar']
        if np.sign(final_pos) != sig_dict['momentum_signal'] and sig_dict['momentum_signal'] != 0:
            final_pos = 0.0
            
        signals[t] = exec_engine.process_signal(final_pos, y, kf.x[0], sig_dict['spread_signal'])
    
    bt = Backtester(taker_fee_bps=0.0)
    is_df = df.iloc[:split_idx].copy()
    res = bt.run(is_df, pd.Series(signals))
    metrics = Backtester.calculate_metrics(res, periods_per_year=8760)
    sharpe = metrics['sharpe_ratio']
    
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = (alpha, q_h, var_thresh, spread)
        print(f"New Best: {best_params} -> Sharpe: {sharpe:.2f}, PnL: {metrics['total_net_pnl']*100:.2f}%")

print(f"Done. Best IS Sharpe: {best_sharpe:.2f} with params: {best_params}")
