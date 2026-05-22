import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.synthetic import MarketSimulator
from src.filters.kalman import AdaptiveKalmanFilter

def calculate_rmse(predictions, targets):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((predictions - targets) ** 2))

def run_benchmark():
    # 1. Generate Ground Truth Data
    print("Generating synthetic data...")
    params = {
        0: {'theta': 0.5, 'mu': 100.0, 'sigma': 5.0},   # Quiet regime, fast mean-reversion, some noise
        1: {'theta': 0.8, 'mu': 115.0, 'sigma': 10.0}   # Volatile regime, faster reversion, large mean shift, high noise
    }
    
    transition_matrix = [
        [0.98, 0.02], 
        [0.05, 0.95]
    ]
    
    sim = MarketSimulator(params, transition_matrix, fat_tail_df=3.0, random_seed=42)
    df = sim.simulate(n_steps=1000, dt=1.0)
    
    prices = df['price'].values
    true_mu = df['true_mu'].values
    
    # 2. Run Adaptive Kalman Filter
    print("Running Adaptive Kalman Filter...")
    # Lower alpha for more stable R estimation, higher Q initialized
    kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.05)
    kf.Q[0, 0] = 0.1
    kf.R[0, 0] = 25.0
    
    kf_means = np.zeros(len(prices))
    kf_stds = np.zeros(len(prices))
    
    for t in range(len(prices)):
        x, P = kf.predict()
        x, P = kf.update(prices[t])
        kf_means[t] = x[0]
        kf_stds[t] = np.sqrt(P[0, 0])
        
    df['kalman_mu'] = kf_means
    df['kalman_std'] = kf_stds
    
    # 3. Baselines: EWMA and 20-period SMA
    print("Computing Baselines...")
    df['sma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
    df['ewma'] = df['price'].ewm(span=20, adjust=False).mean()
    
    # 4. Calculate RMSE
    print("\n--- RMSE Benchmark ---")
    rmse_kalman = calculate_rmse(df['kalman_mu'], true_mu)
    rmse_sma = calculate_rmse(df['sma_20'], true_mu)
    rmse_ewma = calculate_rmse(df['ewma'], true_mu)
    
    print(f"Kalman Filter RMSE: {rmse_kalman:.4f}")
    print(f"20-period SMA RMSE: {rmse_sma:.4f}")
    print(f"EWMA (span=20) RMSE: {rmse_ewma:.4f}")
    
    # 5. Plotting
    print("\nGenerating plot...")
    plt.figure(figsize=(14, 7))
    
    plt.plot(df['time'], df['price'], color='gray', alpha=0.3, label='Observed Price')
    plt.plot(df['time'], true_mu, color='black', linestyle='--', linewidth=2, label='True Fair Value ($\mu$)')
    
    plt.plot(df['time'], df['sma_20'], color='blue', alpha=0.5, label='20-period SMA')
    plt.plot(df['time'], df['ewma'], color='green', alpha=0.5, label='EWMA')
    
    plt.plot(df['time'], df['kalman_mu'], color='red', linewidth=2, label='Kalman Estimate')
    plt.fill_between(df['time'], 
                     df['kalman_mu'] - 1.96 * df['kalman_std'], 
                     df['kalman_mu'] + 1.96 * df['kalman_std'], 
                     color='red', alpha=0.2, label='Kalman 95% Confidence')
    
    plt.title("Fair Value Estimation: Kalman Filter vs Baselines")
    plt.xlabel("Time Step")
    plt.ylabel("Price / Fair Value")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Highlight regimes
    vol_regimes = df[df['true_regime'] == 1]
    for idx, row in vol_regimes.iterrows():
        plt.axvspan(row['time'] - 0.5, row['time'] + 0.5, color='orange', alpha=0.1, lw=0)
        
    plt.tight_layout()
    plt.savefig('kalman_benchmark.png')
    print("Plot saved as 'kalman_benchmark.png'")
    
if __name__ == "__main__":
    run_benchmark()
