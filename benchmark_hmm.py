import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from synthetic import MarketSimulator
from hmm import RegimeHMM

def run_benchmark():
    # 1. Generate Ground Truth Data
    print("Generating synthetic data...")
    # Make regimes very distinct in variance to make detection clear
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},   # Quiet regime (low variance)
        1: {'theta': 0.1, 'mu': 100.0, 'sigma': 4.0}    # Volatile regime (high variance)
    }
    
    # Stay in states for a while so HMM can lock on
    transition_matrix = [
        [0.98, 0.02], 
        [0.05, 0.95]
    ]
    
    sim = MarketSimulator(params, transition_matrix, fat_tail_df=None, random_seed=42)
    df = sim.simulate(n_steps=2000, dt=1.0)
    
    prices = df['price'].values
    true_regime = df['true_regime'].values
    
    # Calculate log returns
    returns = np.zeros(len(prices))
    returns[1:] = np.log(prices[1:] / prices[:-1])
    # For HMM, we will just use standard differences if prices aren't log-normal,
    # but since they mean-revert around 100, differences or log-returns are basically identical.
    # Let's use simple returns to avoid any log(<=0) issues just in case, though they shouldn't happen.
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    
    # 2. Train HMM Offline
    print("Training HMM via Baum-Welch EM...")
    hmm = RegimeHMM(n_states=2, random_seed=42)
    # We fit on returns from index 1 to end
    log_likelihoods = hmm.fit_baum_welch(returns[1:], max_iter=50, tol=1e-4)
    
    print(f"Fitted Volatilities (Quiet, Volatile): {hmm.sigma}")
    print(f"Fitted Transition Matrix:\n{hmm.A}")
    
    # 3. Online Filtering
    print("Running online Forward Filter...")
    # alpha gives P(z_t | y_{1:t})
    alpha = hmm.forward_filter(returns[1:])
    
    # The volatile regime is state 1 (ensured by the sort in fit_baum_welch)
    prob_volatile = np.zeros(len(prices))
    prob_volatile[1:] = alpha[:, 1]
    df['prob_volatile'] = prob_volatile
    
    # 4. Plotting
    print("\nGenerating plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [1.5, 2, 2]})
    
    # Plot 1: Log-Likelihood Convergence
    ax1.plot(log_likelihoods, marker='o', color='blue')
    ax1.set_title("Baum-Welch EM: Log-Likelihood Convergence")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Log-Likelihood")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price and True Regime
    ax2.plot(df['time'], df['price'], color='black', linewidth=1, label='Price')
    
    # Highlight true volatile regimes
    vol_regimes = df[df['true_regime'] == 1]
    for idx, row in vol_regimes.iterrows():
        ax2.axvspan(row['time'] - 0.5, row['time'] + 0.5, color='orange', alpha=0.3, lw=0)
    
    # Add proxy patch for legend
    import matplotlib.patches as mpatches
    orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='True Volatile Regime')
    
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(orange_patch)
    labels.append('True Volatile Regime')
    ax2.legend(handles, labels, loc='upper right')
    
    ax2.set_title("Simulated Price with True Regimes")
    ax2.set_ylabel("Price")
    ax2.set_xlim(df['time'].min(), df['time'].max())
    
    # Plot 3: Online HMM Posterior
    ax3.plot(df['time'], df['prob_volatile'], color='red', label='P(Regime=Volatile | Returns)')
    ax3.fill_between(df['time'], 0, df['prob_volatile'], color='red', alpha=0.2)
    ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Overlay true regime line for comparison
    ax3.plot(df['time'], df['true_regime'], color='orange', linestyle=':', linewidth=2, label='True Regime (0 or 1)')
    
    ax3.set_title("Online HMM Posterior vs True Regime")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Probability")
    ax3.set_xlim(df['time'].min(), df['time'].max())
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('hmm_benchmark.png')
    print("Plots saved as 'hmm_benchmark.png'")
    
if __name__ == "__main__":
    run_benchmark()
