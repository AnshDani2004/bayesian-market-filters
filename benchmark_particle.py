import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from src.filters.kalman import AdaptiveKalmanFilter
from src.filters.particle import StochasticVolParticleFilter

def run_benchmark():
    # 1. Simulate Fat-Tail Event
    print("Simulating fat-tail shock...")
    T = 100
    prices = np.full(T, 100.0)
    
    # Stable period
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(T) * 0.5
    prices[:50] += np.cumsum(noise[:50])
    
    # FAT TAIL SHOCK at t=50 (massive drop)
    prices[50] = prices[49] - 15.0
    
    # Volatile recovery
    prices[51:] = prices[50] + np.cumsum(rng.standard_normal(T-51) * 2.0)
    
    # 2. Run Filters
    kf = AdaptiveKalmanFilter(dt=1.0, initial_price=100.0, adaptive=True, alpha=0.1)
    
    # Higher R prevents all weights going to exactly 0 during a massive shock
    # Higher Q_h allows log-volatility to spike quickly
    pf = StochasticVolParticleFilter(n_particles=10000, initial_price=100.0, Q_h=1.0, random_seed=42)
    
    # We want to capture the posterior EXACTLY at t=50 (the shock)
    shock_idx = 50
    
    for t in range(shock_idx + 1):
        y = prices[t]
        
        # Kalman
        kf.predict()
        kf.update(y)
        
        # Particle
        pf.step(y)
        
    print(f"--- At Shock (t={shock_idx}) ---")
    print(f"Observation: {prices[shock_idx]:.2f}")
    
    # Kalman Posterior
    kf_mean = kf.x[0]
    kf_std = np.sqrt(kf.P[0, 0])
    print(f"Kalman Posterior: Mean={kf_mean:.2f}, Std={kf_std:.2f}")
    
    # Particle Posterior
    pf_mu_particles = pf.mu
    pf_weights = pf.weights
    pf_mean = np.average(pf_mu_particles, weights=pf_weights)
    pf_std = np.sqrt(np.average((pf_mu_particles - pf_mean)**2, weights=pf_weights))
    
    # Calculate Skewness of particles (unweighted for simplicity since we resampled)
    # If weights aren't uniform, we can just use the empirical distribution or resample it to calculate skew
    if pf.get_ess() < pf.N * 0.99:
        # Resample to get unweighted distribution for skew calculation
        positions = (rng.random() + np.arange(pf.N)) / pf.N
        indexes = np.zeros(pf.N, dtype=int)
        cumulative_sum = np.cumsum(pf_weights)
        i, j = 0, 0
        while i < pf.N and j < pf.N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        samples = pf_mu_particles[indexes]
    else:
        samples = pf_mu_particles
        
    pf_skew = skew(samples)
    print(f"Particle Posterior: Mean={pf_mean:.2f}, Std={pf_std:.2f}, Skewness={pf_skew:.4f}")
    
    # 3. The Killer Comparison Plot
    print("\nGenerating 'Killer Comparison' plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot Kalman Gaussian
    x_axis = np.linspace(min(prices[shock_idx]-10, kf_mean-4*kf_std), 
                         max(prices[shock_idx]+10, kf_mean+4*kf_std), 1000)
    kf_pdf = norm.pdf(x_axis, kf_mean, kf_std)
    plt.plot(x_axis, kf_pdf, color='blue', linewidth=2, label=f'Kalman Filter (Gaussian)\nMean: {kf_mean:.1f}')
    plt.fill_between(x_axis, 0, kf_pdf, color='blue', alpha=0.1)
    
    # Plot Particle Filter KDE / Histogram
    plt.hist(samples, bins=50, density=True, color='red', alpha=0.5, 
             label=f'Particle Filter (Full Posterior)\nMean: {pf_mean:.1f}, Skew: {pf_skew:.2f}')
    
    # True Observation
    plt.axvline(prices[shock_idx], color='black', linestyle='--', linewidth=2, 
                label=f'Observed Shock Price: {prices[shock_idx]:.1f}')
    
    plt.title("Posterior Distribution During Fat-Tail Shock: Particle Filter vs Kalman Filter")
    plt.xlabel("Latent Fair Value ($\mu$)")
    plt.ylabel("Probability Density")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('particle_benchmark.png')
    print("Plot saved as 'particle_benchmark.png'")

if __name__ == "__main__":
    run_benchmark()
