# Algorithmic Trading Filters

A collection of sophisticated algorithmic trading filters and regime-detection models, benchmarked against naive moving averages on synthetic market data.

## Tier 1 Features
- **Ground-Truth Data Generator (`synthetic.py`)**: Simulates a hidden Ornstein-Uhlenbeck (OU) fair value process with Markov Chain regime switching and fat-tailed noise.
- **Adaptive Kalman Filter (`kalman.py`)**: Estimates latent fair value from noisy price observations. Automatically tunes its $Q$ and $R$ noise matrices online using a recursive EM update (Robbins-Monro stochastic approximation) to adapt to sudden volatility shocks.
- **Hidden Markov Model (`hmm.py`)**: A 2-state unsupervised regime detector. Includes a Baum-Welch EM algorithm for offline parameter calibration and a Forward filter for real-time posterior probability estimation ($P(\text{regime} \mid \text{data})$).

## Performance Benchmarks

The Adaptive Kalman Filter was evaluated against an Exponentially Weighted Moving Average (EWMA) and a Simple Moving Average (SMA) during a simulated market shock (mean and variance shift). Because the Kalman filter adaptively widened its confidence bounds upon detecting the shock, it re-converged to the true fair value significantly faster than the static moving averages.

| Filter / Estimator | RMSE vs. Ground Truth |
|--------------------|-----------------------|
| **Adaptive Kalman Filter** | **4.8550** |
| EWMA (span=20) | 5.0500 |
| 20-period SMA | 5.7539 |

## Notebooks & Mathematical Derivations
Please review the Jupyter Notebooks for step-by-step mathematical derivations of the state-space models and E-M update loops:
- `notebook_01_kalman.ipynb`: Kalman Filter derivations and expanding confidence band plots.
- `notebook_02_hmm.ipynb`: Forward-Backward EM derivations and real-time regime detection plots.
