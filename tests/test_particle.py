import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from particle import StochasticVolParticleFilter

def test_ess_bounds():
    """
    Test that the Effective Sample Size (ESS) behaves correctly:
    1. It drops when weights become unequal (after an extreme update).
    2. It is restored to N after resampling.
    """
    pf = StochasticVolParticleFilter(n_particles=1000, initial_price=100.0, random_seed=42)
    
    # Initially, weights are uniform, so ESS should be exactly N
    assert np.isclose(pf.get_ess(), 1000.0)
    
    # Predict step (diffuses particles)
    pf.predict()
    
    # Suppose an extreme observation occurs, making weights highly unequal
    extreme_y = 150.0 
    pf.update(extreme_y)
    
    ess_before = pf.get_ess()
    
    # ESS should have dropped significantly below N
    assert ess_before < 1000.0
    
    # If ESS is still > N/2, we force it lower by doing another extreme update
    # just to ensure resampling triggers
    while pf.get_ess() >= 500.0:
        pf.predict()
        pf.update(extreme_y)
        
    ess_before_resample = pf.get_ess()
    assert ess_before_resample < 500.0
    
    # Now trigger resample
    pf.resample()
    
    # After resampling, weights are reset to uniform, so ESS must be back to N
    ess_after = pf.get_ess()
    assert np.isclose(ess_after, 1000.0)

def test_stochastic_vol_predict():
    """
    Test that the predict step properly diffuses both the latent price and the log-volatility.
    """
    pf = StochasticVolParticleFilter(n_particles=1000, initial_price=100.0, Q_h=1.0, random_seed=42)
    
    # Before predict, all h are 0 and mu are 100
    assert np.all(pf.h == 0.0)
    assert np.all(pf.mu == 100.0)
    
    pf.predict()
    
    # After predict, variance of h should be approximately Q_h = 1.0
    var_h = np.var(pf.h)
    assert np.isclose(var_h, 1.0, rtol=0.2)
    
    # The particles of mu should have spread out
    var_mu = np.var(pf.mu)
    assert var_mu > 0.0
