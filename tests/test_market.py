import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.data.market import Backtester

def test_fee_deduction():
    """
    Test that the 0.05% taker fee is correctly mathematically deducted from PnL
    whenever the position size changes.
    """
    bt = Backtester(taker_fee_bps=5.0) # 0.05%
    
    # 4 time steps
    df = pd.DataFrame({
        'close': [100.0, 101.0, 102.0, 102.0]
    })
    
    # Signal: Buy at t=0, Hold at t=1, Close at t=2
    signals = pd.Series([1.0, 1.0, 0.0, 0.0])
    
    results = bt.run(df, signals)
    
    # current_position is signals.shift(1).fillna(0)
    # t=0: pos = 0
    # t=1: pos = 1 (from sig[0]) -> position change = 1 -> fee = 0.0005
    #      asset return = 1% (100 -> 101)
    #      gross return = 1%
    #      net return = 0.01 - 0.0005 = 0.0095
    assert results['current_position'].iloc[1] == 1.0
    assert np.isclose(results['position_change'].iloc[1], 1.0)
    assert np.isclose(results['fee_drag'].iloc[1], 0.0005)
    assert np.isclose(results['gross_return'].iloc[1], 0.01)
    assert np.isclose(results['net_return'].iloc[1], 0.0095)
    
    # t=2: pos = 1 (from sig[1]) -> position change = 0 -> fee = 0
    #      asset return = ~0.99% (101 -> 102)
    #      gross return = 1/101
    assert results['current_position'].iloc[2] == 1.0
    assert np.isclose(results['position_change'].iloc[2], 0.0)
    assert np.isclose(results['fee_drag'].iloc[2], 0.0)
    assert np.isclose(results['gross_return'].iloc[2], 1.0 / 101.0)
    
    # t=3: pos = 0 (from sig[2]) -> position change = 1 -> fee = 0.0005
    #      asset return = 0% (102 -> 102)
    assert results['current_position'].iloc[3] == 0.0
    assert np.isclose(results['position_change'].iloc[3], 1.0)
    assert np.isclose(results['fee_drag'].iloc[3], 0.0005)
    assert np.isclose(results['gross_return'].iloc[3], 0.0)
    assert np.isclose(results['net_return'].iloc[3], -0.0005)
