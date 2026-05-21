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
