"""Audit diagnostics (2026-10-12). Read-only: does not change any strategy code.

Rebuilds the exact signal paths of notebook_03_backtest.py (Alpha Threshold) and
notebook_04_ml_overlay.py (ML overlay) on the frozen CSV, checks they reproduce the
README table, then reports sensitivities a reviewer would ask for:
  1. cost per unit of position change (0, 1.5, 5, 10 bps),
  2. one extra bar of execution delay,
  3. exposure, turnover and long/short split of OOS P&L,
  4. buy-and-hold over the same halves,
  5. Sharpe uncertainty (iid approximation and a moving-block bootstrap).
Run from the repo root:  python audit/audit_diagnostics.py
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.getcwd()))
from src.data.market import Backtester
from src.filters.kalman import AdaptiveKalmanFilter
from src.filters.hmm import RegimeHMM
from src.signals.generator import SignalGenerator
from src.signals.ml_generator import MLSignalGenerator
from src.execution.engine import ExecutionEngine

PPY = 8760
DATA = 'data/btc_1h_2025_2026.csv'
print("data sha256:", hashlib.sha256(open(DATA, 'rb').read()).hexdigest())
df = pd.read_csv(DATA, parse_dates=['timestamp'])
prices = df['close'].values
returns = df['close'].pct_change().fillna(0).values
split_idx = len(prices) // 2
print(f"rows={len(df)} split_idx={split_idx} IS={df.timestamp.iloc[0]}..{df.timestamp.iloc[split_idx-1]} "
      f"OOS={df.timestamp.iloc[split_idx]}..{df.timestamp.iloc[-1]}")
print("volume==0 rows:", int((df['volume'] == 0).sum()))

hmm = RegimeHMM(n_states=2, random_seed=42)
hmm.fit_baum_welch(returns[:split_idx], max_iter=20, tol=1e-3)
p_vol = hmm.forward_filter(returns)[:, 1]
sma_50 = pd.Series(prices).rolling(50).mean().fillna(prices[0]).values
macro = np.where(prices > sma_50, 1, -1)
macro[:50] = 0
eng = ExecutionEngine(taker_fee_bps=5.0)

# --- Alpha Threshold path (notebook_03 lines 54-107; particle filter omitted: its output is unused) ---
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
sg = SignalGenerator(base_spread=0.0005, variance_threshold=500.0)
sig_thr = np.zeros(len(prices))
for t in range(len(prices)):
    y = prices[t]
    kf.predict(); kf.update(y)
    d = sg.generate_signals(mid_price=y, kf_mean=kf.x[0], kf_std=np.sqrt(kf.P[0, 0]),
                            kf_drift=kf.x[1], prob_quiet=1.0 - p_vol[t], macro_trend=macro[t])
    fp = np.clip(d['fair_value_signal'], -1.0, 1.0) * d['position_scalar']
    if np.sign(fp) != d['momentum_signal'] and d['momentum_signal'] != 0:
        fp = 0.0
    sig_thr[t] = eng.process_signal(fp, y, kf.x[0], d['spread_signal'])

# --- ML path (notebook_04 lines 50-118) ---
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
X, Y = [], []
for t in range(len(prices) - 1):
    c = prices[t]
    kf.predict(); kf.update(c)
    s = np.sqrt(kf.P[0, 0])
    X.append([(kf.x[0] - c) / max(s, 1e-6), s / c, 1.0 - p_vol[t], macro[t]])
    Y.append(1 if prices[t + 1] / c - 1.0 > 0 else 0)
X, Y = np.array(X), np.array(Y)
ml = MLSignalGenerator(confidence_threshold=0.55)
ml.train(X[:split_idx - 1], Y[:split_idx - 1])
kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.001)
sig_ml = np.zeros(len(prices))
for t in range(len(prices)):
    c = prices[t]
    kf.predict(); kf.update(c)
    s = np.sqrt(kf.P[0, 0])
    f = np.array([(kf.x[0] - c) / max(s, 1e-6), s / c, 1.0 - p_vol[t], macro[t]])
    d = ml.generate_signals(f, base_spread=0.0005)
    sig_ml[t] = eng.process_signal(d['target_position'], c, kf.x[0], d['spread_signal'])


def metrics(sig, fee_bps, extra_lag=0):
    s = pd.Series(sig).shift(extra_lag).fillna(0) if extra_lag else pd.Series(sig)
    r = Backtester(taker_fee_bps=fee_bps).run(df, s)
    period = np.where(np.arange(len(r)) < split_idx, 'IS', 'OOS')
    return {p: Backtester.calculate_metrics(r[period == p], periods_per_year=PPY) for p in ('IS', 'OOS')}, r, period


def fmt(m):
    return (f"Sharpe {m['sharpe_ratio']:+.2f}  MaxDD {m['max_drawdown']*100:.2f}%  "
            f"Hit {m['hit_rate']*100:.2f}%  NetPnL {m['total_net_pnl']*100:+.2f}%")


print("\n[1] Reproduction of README rows (README fee assumptions: maker 1.5 bps)")
for name, sig in (('AlphaThreshold', sig_thr), ('ML', sig_ml)):
    m, _, _ = metrics(sig, 1.5)
    print(f"  {name:15s} IS : {fmt(m['IS'])}")
    print(f"  {name:15s} OOS: {fmt(m['OOS'])}")

print("\n[2] Cost per unit of position change (bps). 0 = frictionless, 1.5 = README maker, "
      "5 = README taker, 10 = taker plus an assumed 5 bps half-spread")
for name, sig in (('AlphaThreshold', sig_thr), ('ML', sig_ml)):
    for fee in (0.0, 1.5, 5.0, 10.0):
        m, _, _ = metrics(sig, fee)
        print(f"  {name:15s} fee={fee:4.1f}  OOS {fmt(m['OOS'])}")

print("\n[3] One extra bar of execution delay (trade at the next hourly close), fee 1.5 bps")
for name, sig in (('AlphaThreshold', sig_thr), ('ML', sig_ml)):
    m, _, _ = metrics(sig, 1.5, extra_lag=1)
    print(f"  {name:15s} OOS {fmt(m['OOS'])}")

print("\n[4] OOS exposure, turnover, long/short split (fee 1.5 bps)")
for name, sig in (('AlphaThreshold', sig_thr), ('ML', sig_ml)):
    _, r, period = metrics(sig, 1.5)
    o = r[period == 'OOS']
    pos = o['current_position']
    print(f"  {name:15s} hours in market {int((pos != 0).sum())}/{len(o)} "
          f"(long {int((pos > 0).sum())}, short {int((pos < 0).sum())}); "
          f"position changes {int((o['position_change'] > 0).sum())}; "
          f"sum gross {o['gross_return'].sum()*100:+.2f}%, sum fees {o['fee_drag'].sum()*100:.2f}%; "
          f"sum gross from longs {o.loc[pos > 0, 'gross_return'].sum()*100:+.2f}%, "
          f"from shorts {o.loc[pos < 0, 'gross_return'].sum()*100:+.2f}%")

print("\n[5] Buy and hold BTC over the same halves (no fees)")
_, r, period = metrics(np.ones(len(prices)), 0.0)
for p in ('IS', 'OOS'):
    a = r.loc[period == p, 'asset_return'].fillna(0)
    print(f"  {p}: total {((1 + a).prod() - 1)*100:+.2f}%  Sharpe {a.mean()/a.std()*np.sqrt(PPY):+.2f}")

print("\n[6] OOS Sharpe uncertainty (fee 1.5 bps). iid SE ~ sqrt(PPY/N)*sqrt(1+SR_h^2/2); "
      "moving-block bootstrap, block 24 h, 2000 reps, seed 0")
rng = np.random.default_rng(0)
for name, sig in (('AlphaThreshold', sig_thr), ('ML', sig_ml)):
    _, r, period = metrics(sig, 1.5)
    x = r.loc[period == 'OOS', 'net_return'].dropna().values
    n = len(x)
    sr_h = x.mean() / x.std(ddof=1)
    se = np.sqrt(PPY / n) * np.sqrt(1 + 0.5 * sr_h ** 2)
    B, nb = 24, int(np.ceil(n / 24))
    boots = []
    for _ in range(2000):
        starts = rng.integers(0, n - B + 1, nb)
        xb = np.concatenate([x[s:s + B] for s in starts])[:n]
        boots.append(xb.mean() / xb.std(ddof=1) * np.sqrt(PPY))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {name:15s} N={n} Sharpe {sr_h*np.sqrt(PPY):+.2f}  iid SE {se:.2f}  "
          f"t~{sr_h*np.sqrt(PPY)/se:.2f}  block-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}]")
