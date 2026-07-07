import numpy as np
import pandas as pd
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
sma_50 = df['close'].rolling(50).mean().fillna(prices[0]).values
macro_trend = np.where(prices > sma_50, 1, -1)
macro_trend[:50] = 0

split_idx = len(prices) // 2

hmm = RegimeHMM(n_states=2, random_seed=42)
hmm.fit_baum_welch(returns[:split_idx], max_iter=20, tol=1e-3)
prob_volatile_array = hmm.forward_filter(returns)[:, 1]

kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
pf = StochasticVolParticleFilter(n_particles=1000, initial_price=prices[0], Q_h=0.0001)
signal_gen = SignalGenerator(base_spread=0.0005, variance_threshold=500.0)
exec_engine = ExecutionEngine(taker_fee_bps=5.0)

signals = np.zeros(len(prices))
for t in range(len(prices)):
    y = prices[t]
    kf.predict()
    kf.update(y)
    prob_quiet = 1.0 - prob_volatile_array[t]
    
    # generate signals
    std_safe = max(np.sqrt(kf.P[0,0]), 1e-6)
    raw_fv = (kf.x[0] - y) / std_safe
    fv_sig = np.clip(raw_fv, -5.0, 5.0)
    
    # macro trend block
    if macro_trend[t] != 0 and np.sign(fv_sig) != np.sign(macro_trend[t]):
        fv_sig = 0.0
        
    pos_scalar = 0.5 if prob_quiet < 0.5 else 1.0
    final_pos = fv_sig * pos_scalar
    
    if np.sign(final_pos) != np.sign(kf.x[1]) and np.sign(kf.x[1]) != 0:
        final_pos = 0.0
        
    spread_sig = 0.0005 * 1.5 if (kf.P[0,0] > 500.0) else 0.0005
    
    signals[t] = exec_engine.process_signal(final_pos, y, kf.x[0], spread_sig)

bt = Backtester(taker_fee_bps=0.0)
res = bt.run(df, pd.Series(signals))

is_res = res.iloc[:split_idx]
oos_res = res.iloc[split_idx:]
m_is = Backtester.calculate_metrics(is_res, periods_per_year=8760)
m_oos = Backtester.calculate_metrics(oos_res, periods_per_year=8760)

print(f"IS Sharpe: {m_is['sharpe_ratio']:.2f}, PnL: {m_is['total_net_pnl']*100:.2f}%")
print(f"OOS Sharpe: {m_oos['sharpe_ratio']:.2f}, PnL: {m_oos['total_net_pnl']*100:.2f}%")
