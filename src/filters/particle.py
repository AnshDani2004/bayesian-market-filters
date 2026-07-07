import numpy as np
from scipy.stats import norm

class StochasticVolParticleFilter:
    """
    Particle Filter for Stochastic Volatility model.
    Tracks both latent price (mu) and log-volatility (h).
    """
    
    def __init__(self, 
                 n_particles: int = 5000, 
                 initial_price: float = 100.0,
                 dt: float = 1.0,
                 Q_h: float = 0.1,    # Variance of log-volatility random walk
                 random_seed: int = 42):
        self.N = n_particles
        self.dt = dt
        self.Q_h = Q_h
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize particles
        self.mu = np.full(self.N, initial_price)
        # Initialize log-volatility to log(1.0) = 0
        self.h = np.zeros(self.N)
        
        # Initialize weights uniformly
        self.weights = np.ones(self.N) / self.N
        
    def predict(self):
        """
        Predicts the next state for all particles.
        """
        # 1. Diffuse log-volatility
        # h_t = h_{t-1} + N(0, Q_h)
        self.h += self.rng.normal(0, np.sqrt(self.Q_h), self.N)
        
        # Calculate current variance for price diffusion
        # var = exp(h_t)
        price_var = np.exp(self.h)
        
        # 2. Diffuse latent price
        # mu_t = mu_{t-1} + N(0, price_var * dt)
        self.mu += self.rng.normal(0, np.sqrt(price_var * self.dt), self.N)
        
    def update(self, y: float):
        """
        Updates particle weights based on observation y.
        """
        # Calculate likelihood of observation y given each particle's mu and stochastic vol
        # y ~ N(mu, exp(h))
        likelihood = norm.pdf(y, loc=self.mu, scale=np.sqrt(np.exp(self.h)))
        
        # Update weights
        self.weights *= likelihood
        
        # Normalize weights
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            # If all weights go to zero (extreme outlier), reinitialize uniformly
            self.weights = np.ones(self.N) / self.N
            
    def get_ess(self) -> float:
        """Calculates Effective Sample Size (ESS)."""
        # Add small epsilon to prevent division by zero just in case
        return 1.0 / (np.sum(self.weights**2) + 1e-12)
        
    def resample(self):
        """
        Sequential Importance Resampling (SIR).
        Resamples particles if ESS drops below N/2.
        """
        ess = self.get_ess()
        if ess < self.N / 2.0:
            # Systematic resampling
            positions = (self.rng.random() + np.arange(self.N)) / self.N
            indexes = np.zeros(self.N, dtype=int)
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < self.N and j < self.N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
                    
            # Update particles
            self.mu = self.mu[indexes]
            self.h = self.h[indexes]
            
            # Reset weights
            self.weights = np.ones(self.N) / self.N
            
    def step(self, y: float):
        """Runs predict, update, and resample steps."""
        self.predict()
        self.update(y)
        self.resample()
        
    def estimate(self):
        """Returns the weighted mean of the latent price and log-volatility."""
        mean_mu = np.average(self.mu, weights=self.weights)
        mean_h = np.average(self.h, weights=self.weights)
        return mean_mu, mean_h
        
    def get_variance(self) -> float:
        """Returns the estimated variance of the latent price."""
        _, mean_h = self.estimate()
        return np.exp(mean_h)
