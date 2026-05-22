import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.filters.kalman import AdaptiveKalmanFilter
from src.data.synthetic import MarketSimulator

def test_kalman_convergence():
    """
    Test that the Kalman posterior mean converges to the true underlying mean of the OU process.
    """
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},
        1: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0}
    }
    transition_matrix = [[1.0, 0.0], [0.0, 1.0]]
    sim = MarketSimulator(
        params=params,
        transition_matrix=transition_matrix,
        random_seed=42
    )
    df = sim.simulate(n_steps=1000)
    prices = df['price'].values
    
    kf = AdaptiveKalmanFilter(dt=1.0, initial_price=prices[0], adaptive=True, alpha=0.01)
    
    for y in prices:
        kf.predict()
        kf.update(y)
        
    # The true underlying mean is 100.0
    # The posterior should be close to 100.0 after 1000 steps
    assert abs(kf.x[0] - 100.0) < 5.0
