import numpy as np

class SignalGenerator:
    """
    Synthesizes raw posterior outputs from the Kalman Filter, HMM, and Particle Filter
    into actionable trading signals and position sizing multipliers.
    """
    def __init__(self, base_spread: float = 0.05, variance_threshold: float = 2.0):
        """
        Args:
            base_spread: The default quoting spread in normal conditions.
            variance_threshold: The variance level above which the spread is widened.
        """
        self.base_spread = base_spread
        self.variance_threshold = variance_threshold
        
    def generate_signals(self, 
                         mid_price: float, 
                         kf_mean: float, 
                         kf_std: float, 
                         kf_drift: float, 
                         prob_quiet: float) -> dict:
        """
        Generates trading signals for the current timestep.
        
        Args:
            mid_price: Current market mid-price.
            kf_mean: Kalman Filter posterior mean (latent fair value).
            kf_std: Kalman Filter posterior standard deviation.
            kf_drift: Kalman Filter posterior drift.
            prob_quiet: HMM posterior probability of being in a "quiet" regime.
            
        Returns:
            dict containing fair_value_signal, momentum_signal, spread_signal, and position_scalar.
        """
        
        # 1. Fair Value Signal (Continuous z-score)
        # Bounded to [-5, 5] to prevent exploding signals if std is near zero
        std_safe = max(kf_std, 1e-6)
        raw_fv_signal = (kf_mean - mid_price) / std_safe
        fair_value_signal = np.clip(raw_fv_signal, -5.0, 5.0)
        
        # 2. Momentum Signal
        # Simple sign of the drift to act as a trend-following overlay
        momentum_signal = np.sign(kf_drift)
        
        # 3. Spread Signal
        # Widen spread by 1.5x if posterior variance is high
        variance = kf_std ** 2
        if variance > self.variance_threshold:
            spread_signal = self.base_spread * 1.5
        else:
            spread_signal = self.base_spread
            
        # 4. Regime Sizing Signal
        # Halve position size if we are likely in a volatile regime
        if prob_quiet < 0.5:
            position_scalar = 0.5
        else:
            position_scalar = 1.0
            
        return {
            'fair_value_signal': fair_value_signal,
            'momentum_signal': momentum_signal,
            'spread_signal': spread_signal,
            'position_scalar': position_scalar
        }
