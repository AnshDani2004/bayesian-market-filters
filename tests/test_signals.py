import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from signals.generator import SignalGenerator

def test_regime_sizing():
    """Test that position size is halved during volatile regimes."""
    gen = SignalGenerator()
    
    # Quiet regime (prob_quiet > 0.5)
    sig_quiet = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.0, prob_quiet=0.8)
    assert sig_quiet['position_scalar'] == 1.0
    
    # Volatile regime (prob_quiet < 0.5)
    sig_volatile = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.0, prob_quiet=0.2)
    assert sig_volatile['position_scalar'] == 0.5

def test_spread_widening():
    """Test that spread widens by 1.5x when variance exceeds threshold."""
    gen = SignalGenerator(base_spread=0.05, variance_threshold=2.0)
    
    # Low variance (std=1.0, var=1.0 < 2.0)
    sig_low = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.0, prob_quiet=0.8)
    assert np.isclose(sig_low['spread_signal'], 0.05)
    
    # High variance (std=2.0, var=4.0 > 2.0)
    sig_high = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=2.0, kf_drift=0.0, prob_quiet=0.8)
    assert np.isclose(sig_high['spread_signal'], 0.05 * 1.5)

def test_fair_value_signal():
    """Test directional bias of the fair value signal."""
    gen = SignalGenerator()
    
    # Underpriced: mid_price < kf_mean
    sig_buy = gen.generate_signals(mid_price=98.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.0, prob_quiet=0.8)
    assert sig_buy['fair_value_signal'] > 0.0  # Should be +2.0
    
    # Overpriced: mid_price > kf_mean
    sig_sell = gen.generate_signals(mid_price=102.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.0, prob_quiet=0.8)
    assert sig_sell['fair_value_signal'] < 0.0 # Should be -2.0

def test_momentum_signal():
    """Test momentum signal aligns with drift sign."""
    gen = SignalGenerator()
    
    sig_up = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=1.0, kf_drift=0.5, prob_quiet=0.8)
    assert sig_up['momentum_signal'] == 1.0
    
    sig_down = gen.generate_signals(mid_price=100.0, kf_mean=100.0, kf_std=1.0, kf_drift=-0.5, prob_quiet=0.8)
    assert sig_down['momentum_signal'] == -1.0
