import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.data.synthetic import MarketSimulator

def test_ou_convergence():
    """
    Test that without regime switching, the generated OU process converges 
    to the theoretical mean and variance.
    """
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},
        1: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0}
    }
    
    transition_matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    sim = MarketSimulator(params, transition_matrix, fat_tail_df=None, random_seed=42)
    df = sim.simulate(n_steps=100000, dt=1.0)
    
    # We drop the first 5000 steps as burn-in
    burn_in = 5000
    prices = df['price'].values[burn_in:]
    
    sample_mean = np.mean(prices)
    sample_var = np.var(prices)
    
    theoretical_mean = 100.0
    # True variance of OU process is sigma^2 / (2 * theta)
    theoretical_var = (1.0 ** 2) / (2 * 0.1)
    
    assert np.isclose(sample_mean, theoretical_mean, rtol=0.01), f"Mean {sample_mean} != {theoretical_mean}"
    assert np.isclose(sample_var, theoretical_var, rtol=0.1), f"Var {sample_var} != {theoretical_var}"

def test_regime_switching():
    """
    Test that the empirical regime switching proportions match the 
    theoretical stationary distribution.
    """
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},
        1: {'theta': 0.05, 'mu': 90.0, 'sigma': 3.0}
    }
    
    transition_matrix = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    
    sim = MarketSimulator(params, transition_matrix, fat_tail_df=None, random_seed=42)
    df = sim.simulate(n_steps=100000)
    
    regime_counts = df['true_regime'].value_counts(normalize=True)
    
    # Stationary distribution pi * P = pi
    # pi_0 = 0.9 * pi_0 + 0.2 * pi_1
    # 0.1 * pi_0 = 0.2 * pi_1 => pi_0 = 2 * pi_1
    # pi_0 + pi_1 = 1 => 3 * pi_1 = 1 => pi_1 = 1/3, pi_0 = 2/3
    theoretical_pi_0 = 2/3
    theoretical_pi_1 = 1/3
    
    assert np.isclose(regime_counts[0], theoretical_pi_0, atol=0.05), f"pi_0 {regime_counts[0]} != {theoretical_pi_0}"
    assert np.isclose(regime_counts[1], theoretical_pi_1, atol=0.05), f"pi_1 {regime_counts[1]} != {theoretical_pi_1}"

def test_fat_tail():
    """
    Test that fat-tailed noise results in higher kurtosis than Gaussian noise.
    """
    params = {
        0: {'theta': 0.0, 'mu': 100.0, 'sigma': 1.0},
        1: {'theta': 0.0, 'mu': 100.0, 'sigma': 1.0}
    }
    
    transition_matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ])
    
    # Gaussian
    sim_gauss = MarketSimulator(params, transition_matrix, fat_tail_df=None, random_seed=42)
    df_gauss = sim_gauss.simulate(n_steps=10000, dt=1.0)
    diff_gauss = df_gauss['price'].diff().dropna()
    kurtosis_gauss = diff_gauss.kurtosis()
    
    # Fat-tail (df=3 has very heavy tails)
    sim_fat = MarketSimulator(params, transition_matrix, fat_tail_df=3.0, random_seed=42)
    df_fat = sim_fat.simulate(n_steps=10000, dt=1.0)
    diff_fat = df_fat['price'].diff().dropna()
    kurtosis_fat = diff_fat.kurtosis()
    
    # Kurtosis of normal is ~0 (pandas uses Fisher's definition where normal is 0)
    # Fat tail should be significantly > 0
    assert kurtosis_fat > kurtosis_gauss + 1.0, f"Fat tail kurtosis {kurtosis_fat} not > Gaussian {kurtosis_gauss}"
