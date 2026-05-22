import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.filters.hmm import RegimeHMM
from src.data.synthetic import MarketSimulator

def test_hmm_monotonicity():
    """
    Test that the Baum-Welch algorithm log-likelihood is monotonically increasing.
    """
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},
        1: {'theta': 0.1, 'mu': 100.0, 'sigma': 5.0}
    }
    transition_matrix = [[0.95, 0.05], [0.05, 0.95]]
    sim = MarketSimulator(
        params=params,
        transition_matrix=transition_matrix,
        random_seed=42
    )
    df = sim.simulate(n_steps=1000)
    returns = df['price'].pct_change().fillna(0).values
    
    hmm = RegimeHMM(n_states=2, random_seed=42)
    lls = hmm.fit_baum_welch(returns, max_iter=20, tol=1e-6)
    
    # Assert monotonicity with floating-point tolerance
    assert all(b - a > -1e-6 for a, b in zip(lls, lls[1:]))
