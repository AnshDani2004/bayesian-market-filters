import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class MLSignalGenerator:
    """
    Uses a Gradient Boosting Classifier to predict the forward 1-hour return direction 
    based on the state of the mathematical filters.
    """
    def __init__(self, confidence_threshold: float = 0.55):
        """
        Args:
            confidence_threshold: Probability threshold above which the model takes a directional trade.
                                  e.g., if P(Up) > 0.55, go Long. If P(Up) < 0.45, go Short.
        """
        # We use a relatively shallow tree depth to prevent overfitting on the IS data
        self.model = GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=3, 
            random_state=42
        )
        self.confidence_threshold = confidence_threshold
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the Gradient Boosting model on the In-Sample dataset.
        
        Args:
            X: Matrix of shape (n_samples, n_features) containing filter states.
            y: Array of shape (n_samples,) containing binary forward returns (1 for Up, 0 for Down).
        """
        self.model.fit(X, y)
        self.is_trained = True
        
    def generate_signals(self, features: np.ndarray, base_spread: float = 0.0005) -> dict:
        """
        Predicts probability of a positive return and generates an execution signal.
        
        Args:
            features: 1D array of features for the current timestep.
            base_spread: Base quoting spread.
            
        Returns:
            dict containing the target_position, the raw probability, and the quoting spread.
        """
        if not self.is_trained:
            raise ValueError("ML Model must be trained before generating online signals.")
            
        # Predict probability of the positive class (Up)
        prob_up = self.model.predict_proba(features.reshape(1, -1))[0, 1]
        
        # Probabilistic Execution Threshold
        if prob_up > self.confidence_threshold:
            target_pos = 1.0
        elif prob_up < (1.0 - self.confidence_threshold):
            target_pos = -1.0
        else:
            target_pos = 0.0
            
        return {
            'target_position': target_pos,
            'prob_up': prob_up,
            'spread_signal': base_spread
        }
