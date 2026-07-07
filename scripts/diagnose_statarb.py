import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.filters.kalman import PairsKalmanFilter

def run_diagnostics():
    data_file = 'data/btc_eth_1h_2025_2026.csv'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found.")
        return
        
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    
    # Calculate log prices
    df['log_btc'] = np.log(df['btc_close'])
    df['log_eth'] = np.log(df['eth_close'])

    prices_btc = df['log_btc'].values
    prices_eth = df['log_eth'].values

    kf = PairsKalmanFilter(delta=1e-4, R=1e-3)
    
    target_positions = np.zeros(len(df))
    spreads = np.zeros(len(df))
    betas = np.zeros(len(df))
    variances = np.zeros(len(df))
    
    for t in range(len(df)):
        x_price = prices_eth[t]
        y_price = prices_btc[t]
        
        # 1. Update Pairs Kalman Filter
        fair_value_y = kf.step(x_price, y_price)
        
        # Extract volatility from Kalman innovation covariance S
        variances[t] = kf.S[0, 0]
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
        kelly_fraction = np.clip(z_score / 3.0, -1.0, 1.0) 
        if t < 100:
            kelly_fraction = 0.0
            
        target_positions[t] = kelly_fraction

    df['target_position'] = target_positions
    df['kf_beta'] = betas
    df['spread_error'] = spreads
    df['variance'] = variances
    
    df['return_btc'] = df['btc_close'].pct_change().fillna(0)
    df['return_eth'] = df['eth_close'].pct_change().fillna(0)
    df['spread_return'] = df['return_btc'] - df['kf_beta'].shift(1).fillna(1.0) * df['return_eth']
    df['current_position'] = df['target_position'].shift(1).fillna(0)
    df['gross_return'] = df['current_position'] * df['spread_return']
    fee_bps = 1.5 / 10000.0
    df['position_change'] = df['current_position'].diff().fillna(0).abs()
    df['fee_drag'] = df['position_change'] * 2.0 * fee_bps
    df['net_return'] = df['gross_return'] - df['fee_drag']
    df['cum_net'] = (1 + df['net_return']).cumprod() - 1

    split_idx = len(df) // 2
    
    # --- PLOTS ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 25))
    
    # 1. Normalized prices and Spread
    norm_btc = df['btc_close'] / df['btc_close'].iloc[0]
    norm_eth = df['eth_close'] / df['eth_close'].iloc[0]
    ax = axes[0]
    ax.plot(df['timestamp'], norm_btc, label='Normalized BTC', alpha=0.7)
    ax.plot(df['timestamp'], norm_eth, label='Normalized ETH', alpha=0.7)
    ax.set_ylabel('Normalized Price')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(df['timestamp'], df['spread_error'], color='purple', label='Kalman Spread', alpha=0.5)
    ax2.set_ylabel('Spread')
    ax.axvline(df['timestamp'].iloc[split_idx], color='red', linestyle='--', label='IS/OOS Split')
    ax.set_title('Normalized BTC vs ETH & Kalman Spread')
    
    # 2. Hedge Ratio (Beta)
    ax = axes[1]
    ax.plot(df['timestamp'], df['kf_beta'], color='orange', label='Hedge Ratio (Beta)')
    ax.axvline(df['timestamp'].iloc[split_idx], color='red', linestyle='--')
    ax.set_title('Kalman Filter Hedge Ratio (Beta)')
    ax.legend()
    
    # 3. Innovation Variance (S)
    ax = axes[2]
    ax.plot(df['timestamp'], df['variance'], color='red', label='Innovation Variance (S)')
    ax.axvline(df['timestamp'].iloc[split_idx], color='red', linestyle='--')
    ax.set_title('Innovation Variance over Time')
    ax.legend()
    
    # 4. Equity Curve vs Spread
    ax = axes[3]
    ax.plot(df['timestamp'], df['cum_net'] * 100, label='Equity Curve (%)', color='green')
    ax.set_ylabel('Cum Net Return (%)')
    ax.legend(loc='upper left')
    ax2 = ax.twinx()
    ax2.plot(df['timestamp'], df['spread_error'], color='purple', label='Kalman Spread', alpha=0.3)
    ax2.set_ylabel('Spread')
    ax.axvline(df['timestamp'].iloc[split_idx], color='red', linestyle='--')
    ax.set_title('Equity Curve vs Kalman Spread')
    
    # 5. Spread Return vs Fees
    ax = axes[4]
    ax.plot(df['timestamp'], df['gross_return'].cumsum() * 100, label='Gross Return (%)', color='blue')
    ax.plot(df['timestamp'], (df['fee_drag'] * -1).cumsum() * 100, label='Cumulative Fees (%)', color='red')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.axvline(df['timestamp'].iloc[split_idx], color='red', linestyle='--')
    ax.set_title('Gross Return vs Cumulative Fees')
    
    plt.tight_layout()
    plt.savefig('statarb_diagnostic.png')
    
    # --- ADF TEST ---
    print("=== Augmented Dickey-Fuller Test on Kalman Spread ===")
    
    # IS Spread
    spread_is = df['spread_error'].iloc[:split_idx].values
    result_is = adfuller(spread_is[100:]) # ignore warmup
    print(f"In-Sample ADF Statistic: {result_is[0]:.4f}")
    print(f"In-Sample p-value: {result_is[1]:.4f}")
    
    # OOS Spread
    spread_oos = df['spread_error'].iloc[split_idx:].values
    result_oos = adfuller(spread_oos)
    print(f"Out-Of-Sample ADF Statistic: {result_oos[0]:.4f}")
    print(f"Out-Of-Sample p-value: {result_oos[1]:.4f}")

if __name__ == '__main__':
    run_diagnostics()
