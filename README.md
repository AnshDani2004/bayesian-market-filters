# Algorithmic Trading Filters

A collection of sophisticated algorithmic trading filters and regime-detection models, benchmarked against naive moving averages on synthetic market data, and backtested on live Binance 1-minute crypto data.

## Final System Architecture

```mermaid
graph TD
    A[Binance REST API / Live Data] -->|1m OHLCV| B{Filter Stack}
    
    B --> C[Adaptive Kalman Filter]
    B --> D[HMM Regime Detector]
    B --> E[Stochastic Vol Particle Filter]
    
    C -->|Latent Fair Value| F(Signal Generator)
    D -->|P_quiet| F
    E -->|Skewness / Var| F
    
    F -->|Raw Direction & Size| G[Execution Engine]
    G -->|Alpha Threshold| H[Portfolio Tracker]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
```

## Tier 1 Features
- **Ground-Truth Data Generator (`synthetic.py`)**: Simulates a hidden Ornstein-Uhlenbeck (OU) fair value process with Markov Chain regime switching and fat-tailed noise.
- **Adaptive Kalman Filter (`kalman.py`)**: Estimates latent fair value from noisy price observations. Automatically tunes its $Q$ and $R$ noise matrices online using a recursive EM update (Robbins-Monro stochastic approximation) to adapt to sudden volatility shocks.
- **Hidden Markov Model (`hmm.py`)**: A 2-state unsupervised regime detector. Includes a Baum-Welch EM algorithm for offline parameter calibration and a Forward filter for real-time posterior probability estimation ($P(\text{regime} \mid \text{data})$).
- **Particle Filter (`particle.py`)**: Uses Sequential Importance Resampling (SIR) to maintain 10,000 particles tracking unobservable stochastic log-volatility. Unlike Kalman, this properly isolates skewed, fat-tailed downside risk.

## Capstone Backtest Results (Live BTC Data)

The system was evaluated on ~9,000 live 1-minute BTC/USDT candles over a continuous 7-day period (May 16 - May 22, 2026). We executed a comparison between a naive taker-fee model (Continuous Bleed) and our execution-optimized limit-order model utilizing an **Alpha Threshold** (Execution Engine).

| Strategy | Sharpe Ratio | Max Drawdown | Hit Rate | Total Net PnL |
|----------|--------------|--------------|----------|---------------|
| **Naive (Taker, Continuous Bleed)** | -354.00 | -64.74% | < 30% | -64.74% |
| **Alpha Threshold (Maker, Limit)** | **+5.01** | **-0.25%** | **59.52%** | **+0.21%** |

*Note: The Sharpe ratio is annualized based on a 1-minute frequency. We calculate the annualization factor assuming 24/7 crypto markets: $\sqrt{365 \times 24 \times 60} = \sqrt{525,600} \approx 725.0$.*

**A Note on Limitations**: While the mathematical integrity of the execution layer holds firm, this is a single-asset demonstration. The quoted results do not explicitly model multi-asset portfolio constraints, complex multi-level orderbook slippage beyond the base taker fee, or adversarial market impact. In a live environment, the hit rate and net PnL will scale with available liquidity.

## Notebooks & Mathematical Derivations
Please review the Jupyter Notebooks for step-by-step mathematical derivations of the state-space models, E-M update loops, and execution rules:
- `notebook_01_kalman.ipynb`: Kalman Filter derivations and expanding confidence band plots.
- `notebook_02_hmm.ipynb`: Forward-Backward EM derivations and real-time regime detection plots.
- `notebook_03_backtest.ipynb`: Final capstone architecture backtest, live data ingestion, and comparative equity curves.
