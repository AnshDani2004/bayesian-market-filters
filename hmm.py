import numpy as np
from scipy.stats import norm
import warnings

class RegimeHMM:
    """
    2-State Hidden Markov Model for Regime Detection.
    Emissions are modeled as Gaussian distributions.
    States: 0 (Quiet), 1 (Volatile)
    """
    
    def __init__(self, n_states: int = 2, random_seed: int = 42):
        self.n_states = n_states
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize parameters
        # Transition matrix A[i, j] = P(z_t=j | z_{t-1}=i)
        self.A = np.array([
            [0.9, 0.1],
            [0.1, 0.9]
        ])
        
        # Initial state probabilities
        self.pi = np.array([0.5, 0.5])
        
        # Emission parameters (mean, std)
        self.mu = np.zeros(self.n_states)
        self.sigma = np.ones(self.n_states)
        
    def _emission_probs(self, y: np.ndarray) -> np.ndarray:
        """Computes P(y_t | z_t=i) for all t and i."""
        T = len(y)
        B = np.zeros((T, self.n_states))
        for i in range(self.n_states):
            # Add small epsilon to variance to prevent division by zero
            var = max(self.sigma[i]**2, 1e-8)
            std = np.sqrt(var)
            B[:, i] = norm.pdf(y, loc=self.mu[i], scale=std)
            
        # Prevent absolute zeros in emission probs
        B = np.maximum(B, 1e-12)
        return B
        
    def forward_filter(self, y: np.ndarray) -> np.ndarray:
        """
        Online Forward algorithm.
        Returns the filtered probabilities P(z_t | y_{1:t})
        """
        T = len(y)
        alpha = np.zeros((T, self.n_states))
        B = self._emission_probs(y)
        
        # Initialization (t=0)
        alpha[0] = self.pi * B[0]
        alpha[0] /= np.sum(alpha[0])
        
        # Recursion
        for t in range(1, T):
            # Predict
            pred = alpha[t-1] @ self.A
            # Update
            alpha[t] = pred * B[t]
            # Normalize
            norm_factor = np.sum(alpha[t])
            if norm_factor > 0:
                alpha[t] /= norm_factor
            else:
                alpha[t] = np.ones(self.n_states) / self.n_states
                
        return alpha

    def forward_backward(self, y: np.ndarray) -> tuple:
        """
        Scaled Forward-Backward algorithm (E-step).
        Returns:
            gamma: Smoothed probabilities P(z_t | y_{1:T})
            xi: Joint probabilities P(z_t, z_{t+1} | y_{1:T})
            log_likelihood: Log-likelihood of the observation sequence
        """
        T = len(y)
        B = self._emission_probs(y)
        
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        c = np.zeros(T) # Scaling factors
        
        # Forward Pass
        alpha[0] = self.pi * B[0]
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]
        
        for t in range(1, T):
            alpha[t] = (alpha[t-1] @ self.A) * B[t]
            c[t] = np.sum(alpha[t])
            if c[t] == 0:
                c[t] = 1e-12
            alpha[t] /= c[t]
            
        log_likelihood = np.sum(np.log(c))
        
        # Backward Pass
        beta[-1] = 1.0
        for t in range(T-2, -1, -1):
            beta[t] = (self.A @ (B[t+1] * beta[t+1])) / c[t+1]
            
        # Compute Gamma and Xi
        gamma = alpha * beta
        # Normalize gamma just in case
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        xi = np.zeros((T-1, self.n_states, self.n_states))
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * self.A[i, j] * B[t+1, j] * beta[t+1, j]
            xi[t] /= np.sum(xi[t])
            
        return gamma, xi, log_likelihood

    def fit_baum_welch(self, y: np.ndarray, max_iter: int = 100, tol: float = 1e-4) -> list:
        """
        Trains the HMM using the Baum-Welch EM algorithm.
        
        Returns:
            List of log-likelihoods per iteration.
        """
        # Initialize parameters smartly based on data if completely naive
        if np.all(self.mu == 0):
            # Sort by variance: quiet regime has lower variance
            var_guess = np.var(y)
            self.sigma = np.array([np.sqrt(var_guess)*0.5, np.sqrt(var_guess)*2.0])
            self.mu = np.array([np.mean(y), np.mean(y)])
            
        log_likelihoods = []
        
        for iteration in range(max_iter):
            # E-step
            gamma, xi, ll = self.forward_backward(y)
            log_likelihoods.append(ll)
            
            # Check convergence
            if iteration > 0 and abs(ll - log_likelihoods[-2]) < tol:
                break
                
            # M-step
            self.pi = gamma[0]
            
            # Update A
            sum_xi = np.sum(xi, axis=0)
            sum_gamma_T1 = np.sum(gamma[:-1], axis=0)
            for i in range(self.n_states):
                if sum_gamma_T1[i] > 0:
                    self.A[i] = sum_xi[i] / sum_gamma_T1[i]
            # Ensure A rows sum to 1
            self.A /= np.sum(self.A, axis=1, keepdims=True)
            
            # Update Mu and Sigma
            sum_gamma = np.sum(gamma, axis=0)
            for i in range(self.n_states):
                if sum_gamma[i] > 0:
                    self.mu[i] = np.sum(gamma[:, i] * y) / sum_gamma[i]
                    diff = y - self.mu[i]
                    var = np.sum(gamma[:, i] * (diff**2)) / sum_gamma[i]
                    self.sigma[i] = np.sqrt(max(var, 1e-8))
                    
        # Sort states post-fitting to ensure state 0 is 'quiet' and state 1 is 'volatile'
        if self.sigma[0] > self.sigma[1]:
            # Swap states
            self.sigma = self.sigma[::-1]
            self.mu = self.mu[::-1]
            self.pi = self.pi[::-1]
            self.A = self.A[::-1, ::-1]
            
        return log_likelihoods
