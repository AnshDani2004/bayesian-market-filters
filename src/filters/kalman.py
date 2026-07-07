import numpy as np
from typing import Tuple, Optional

class AdaptiveKalmanFilter:
    """
    A Linear Kalman Filter with online adaptive Q and R estimation.
    State vector: x_t = [fair_value, drift]^T
    Observation:  y_t = price
    """
    def __init__(self, 
                 dt: float = 1.0, 
                 initial_price: float = 100.0,
                 adaptive: bool = True,
                 alpha: float = 0.01):
        """
        Args:
            dt: Time step.
            initial_price: Starting observation to initialize the state.
            adaptive: If True, uses recursive online EM to update Q and R.
            alpha: Learning rate for adaptive Q and R updates (e.g., 0.01 ~ 100 period window).
        """
        self.dt = dt
        self.adaptive = adaptive
        self.alpha = alpha
        
        # State: [fair_value, drift]
        self.x = np.array([initial_price, 0.0])
        
        # State covariance matrix P
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 0.1]
        ])
        
        # Transition matrix F
        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ])
        
        # Observation matrix H
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance Q (initialized to small values)
        self.Q = np.array([
            [1e-4, 0.0],
            [0.0, 1e-5]
        ])
        
        # Observation noise covariance R
        self.R = np.array([[1.0]])
        
        # Identity matrix for updates
        self.I = np.eye(2)
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the next state and covariance.
        Returns:
            Tuple of (predicted state, predicted covariance)
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P
        
    def update(self, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Updates the state with a new observation and adaptively estimates Q and R if enabled.
        
        Args:
            y: New observed price.
            
        Returns:
            Tuple of (updated state, updated covariance)
        """
        # Innovation (residual)
        v = y - (self.H @ self.x)[0]
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Adaptive updates for Q and R (recursive EM / Robbins-Monro)
        if self.adaptive:
            # R update: Variance of innovation minus predicted state variance
            # R_new = (1 - alpha) * R_old + alpha * (v*v^T - H*P*H^T)
            # We bound it to be positive definite
            v_sq = v**2
            HPHT = (self.H @ self.P @ self.H.T)[0, 0]
            R_est = max(1e-4, v_sq - HPHT)
            self.R[0, 0] = (1 - self.alpha) * self.R[0, 0] + self.alpha * R_est
            
            # Q update: Based on the state correction K * v
            # Q_new = (1 - alpha) * Q_old + alpha * (K*v * (K*v)^T)
            correction = K * v
            Q_est = correction @ correction.T
            self.Q = (1 - self.alpha) * self.Q + self.alpha * Q_est
            
            # Ensure Q remains diagonal or symmetric positive definite (we'll keep it diagonal for stability)
            self.Q[0, 1] = 0.0
            self.Q[1, 0] = 0.0
            self.Q[0, 0] = max(1e-6, self.Q[0, 0])
            self.Q[1, 1] = max(1e-8, self.Q[1, 1])

        # State update
        self.x = self.x + (K * v).flatten()
        
        # Covariance update (Joseph form for numerical stability)
        IKH = self.I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        
        return self.x, self.P

    def step(self, y: float) -> float:
        """
        Convenience method to run predict and update in one step.
        Returns the posterior fair value estimate.
        """
        self.predict()
        self.update(y)
        return self.x[0]


class PairsKalmanFilter:
    """
    A Kalman Filter for Cointegration / Statistical Arbitrage.
    Tracks the dynamic hedge ratio (beta) and intercept (alpha) between two assets.
    Observation: y_t = alpha_t + beta_t * x_t + v_t
    State: s_t = [alpha_t, beta_t]^T
    """
    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        """
        Args:
            delta: Process noise variance (controls how fast beta/alpha adapt).
            R: Observation noise variance (controls how much we trust the spread).
        """
        self.x = np.array([0.0, 1.0])  # Initialize [alpha, beta]
        self.P = np.eye(2)             # State covariance
        
        self.Vw = delta / (1 - delta) * np.eye(2) # Process noise (Q)
        self.Ve = R                               # Observation noise (R)
        self.I = np.eye(2)
        
    def step(self, x_price: float, y_price: float) -> float:
        """
        Predicts and updates the filter based on the new prices.
        
        Args:
            x_price: Price of independent asset (e.g. ETH)
            y_price: Price of dependent asset (e.g. BTC)
            
        Returns:
            The fair value of y_price based on the cointegration.
        """
        # Predict step
        # State transition is Identity (F = I). Random walk for alpha and beta.
        # x_{t|t-1} = x_{t-1|t-1}
        self.P = self.P + self.Vw
        
        # Observation matrix H = [1, x_price]
        H = np.array([[1.0, x_price]])
        
        # Innovation (residual spread)
        y_pred = (H @ self.x)[0]
        e = y_price - y_pred
        
        # Innovation covariance S
        self.S = H @ self.P @ H.T + self.Ve
        
        # Kalman Gain K
        K = self.P @ H.T @ np.linalg.inv(self.S)
        
        # Update step
        self.x = self.x + (K * e).flatten()
        
        # Covariance update (Joseph form)
        IKH = self.I - K @ H
        self.P = IKH @ self.P @ IKH.T + K * self.Ve @ K.T
        
        # Return updated fair value (using updated state)
        return (H @ self.x)[0]

