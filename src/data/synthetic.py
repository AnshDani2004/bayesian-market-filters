import numpy as np
import pandas as pd
from typing import Dict, Optional

class MarketSimulator:
    """
    Generates synthetic market data using a regime-switching Ornstein-Uhlenbeck process.
    Provides ground truth for testing filtering algorithms (Kalman, HMM, Particle Filters).
    """
    
    def __init__(self, 
                 params: Dict[int, Dict[str, float]], 
                 transition_matrix: np.ndarray,
                 fat_tail_df: Optional[float] = None,
                 random_seed: Optional[int] = None):
        """
        Args:
            params: Dictionary mapping regime state (0, 1) to parameters {'theta', 'mu', 'sigma'}.
            transition_matrix: 2x2 Markov transition matrix.
            fat_tail_df: Degrees of freedom for Student-t distribution. If None, uses Gaussian.
            random_seed: Seed for reproducibility.
        """
        self.params = params
        self.transition_matrix = np.asarray(transition_matrix)
        self.fat_tail_df = fat_tail_df
        self.rng = np.random.default_rng(random_seed)
        
    def _generate_regime_path(self, n_steps: int, initial_state: int = 0) -> np.ndarray:
        """Generates a sequence of states from the Markov chain."""
        states = np.zeros(n_steps, dtype=int)
        states[0] = initial_state
        
        for t in range(1, n_steps):
            current_state = states[t-1]
            probs = self.transition_matrix[current_state]
            states[t] = self.rng.choice([0, 1], p=probs)
            
        return states
        
    def simulate(self, n_steps: int, dt: float = 1.0, initial_price: Optional[float] = None) -> pd.DataFrame:
        """
        Runs the Euler-Maruyama simulation for the regime-switching OU process.
        
        Args:
            n_steps: Number of time steps to simulate.
            dt: Time step size.
            initial_price: Starting price. If None, starts at the mu of the initial regime.
            
        Returns:
            pd.DataFrame: DataFrame containing time, price, and ground truth parameters.
        """
        # 1. Generate the underlying hidden Markov regime path
        regimes = self._generate_regime_path(n_steps)
        
        # 2. Extract parameters for each time step based on the regime
        theta_path = np.array([self.params[state]['theta'] for state in regimes])
        mu_path = np.array([self.params[state]['mu'] for state in regimes])
        sigma_path = np.array([self.params[state]['sigma'] for state in regimes])
        
        # 3. Generate noise increments (dW_t)
        if self.fat_tail_df is not None:
            # Student-t distributed noise scaled to have variance 1 if df > 2
            if self.fat_tail_df > 2:
                scale_factor = np.sqrt((self.fat_tail_df - 2) / self.fat_tail_df)
            else:
                scale_factor = 1.0
            noise = self.rng.standard_t(df=self.fat_tail_df, size=n_steps) * scale_factor
        else:
            noise = self.rng.standard_normal(size=n_steps)
            
        # 4. Euler-Maruyama simulation loop
        prices = np.zeros(n_steps)
        if initial_price is None:
            prices[0] = mu_path[0]
        else:
            prices[0] = initial_price
        
        sqrt_dt = np.sqrt(dt)
        
        for t in range(1, n_steps):
            # Previous state
            S_prev = prices[t-1]
            
            # Current parameters
            theta = theta_path[t]
            mu = mu_path[t]
            sigma = sigma_path[t]
            
            # SDE Discretization: dS = theta * (mu - S) * dt + sigma * dW
            drift = theta * (mu - S_prev) * dt
            diffusion = sigma * sqrt_dt * noise[t]
            
            prices[t] = S_prev + drift + diffusion
            
        # 5. Package into DataFrame
        df = pd.DataFrame({
            'time': np.arange(n_steps) * dt,
            'price': prices,
            'true_regime': regimes,
            'true_mu': mu_path,
            'true_theta': theta_path,
            'true_sigma': sigma_path
        })
        
        return df

if __name__ == "__main__":
    # Example usage / basic test
    params = {
        0: {'theta': 0.1, 'mu': 100.0, 'sigma': 1.0},   # Quiet regime
        1: {'theta': 0.05, 'mu': 90.0, 'sigma': 3.0}    # Volatile regime
    }
    
    transition_matrix = [
        [0.99, 0.01], # Stay in 0 with 99% prob
        [0.05, 0.95]  # Stay in 1 with 95% prob
    ]
    
    sim = MarketSimulator(params, transition_matrix, fat_tail_df=3.0, random_seed=42)
    df = sim.simulate(n_steps=1000)
    
    print(df.head())
    print(f"\nRegime counts:\n{df['true_regime'].value_counts()}")
