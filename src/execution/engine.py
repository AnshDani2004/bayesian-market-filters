import numpy as np

class ExecutionEngine:
    """
    Applies execution logic and the 'Alpha Threshold' to raw trading signals.
    Prevents fee bleed by forcing the strategy to remain flat unless the edge
    exceeds the estimated transaction cost.
    """
    def __init__(self, taker_fee_bps: float = 5.0):
        self.taker_fee = taker_fee_bps / 10000.0
        
    def process_signal(self, 
                       target_position: float, 
                       mid_price: float, 
                       kf_mean: float, 
                       spread_signal: float) -> float:
        """
        Applies the alpha threshold.
        
        Args:
            target_position: The raw target position recommended by the SignalGenerator.
            mid_price: The current market mid-price.
            kf_mean: The latent fair value estimate.
            spread_signal: The dynamically adjusted quote spread (as a percentage).
            
        Returns:
            The executed position. Either the `target_position` or 0.0.
        """
        # Calculate mispricing as a percentage of mid_price
        mispricing_pct = abs(kf_mean - mid_price) / mid_price
        
        # Total execution cost (spread + taker fee) as a percentage
        cost_pct = spread_signal + self.taker_fee
        
        if mispricing_pct > cost_pct:
            return target_position
        else:
            # The edge is gone or too small to cover fees; close the position.
            return 0.0
