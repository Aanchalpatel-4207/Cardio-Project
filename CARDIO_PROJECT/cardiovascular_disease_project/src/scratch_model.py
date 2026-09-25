"""
Cardiovascular Disease Prediction - Custom Logistic Regression from Scratch
Fulfills Darshan University ML Project SOP - Section 3.1 Item 4 (Mandatory Scratch Algorithm Constraint).
Built purely using NumPy vectorization without external ML libraries.
"""

import numpy as np
import pickle

class ScratchLogisticRegression:
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.algorithm_name = "Logistic Regression (Implemented from Scratch)"

    def _sigmoid(self, z):
        """Numerically stable sigmoid activation function."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """
        Trains the Logistic Regression model using Gradient Descent.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        - y: numpy array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        y = np.array(y, dtype=float)

        for i in range(self.n_iterations):
            # Forward pass: linear combination + sigmoid activation
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute Binary Cross-Entropy Loss
            epsilon = 1e-15
            y_pred_clipped = np.clip(y_predicted, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

            # Compute Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Gradient Descent Weight Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        Returns array of shape (n_samples, 2) where col 0 = P(y=0), col 1 = P(y=1).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        prob_positive = self._sigmoid(linear_model)
        prob_negative = 1.0 - prob_positive
        return np.column_stack((prob_negative, prob_positive))

    def predict(self, X, threshold=0.5):
        """Predicts binary outcome {0, 1} for X."""
        probs = self.predict_proba(X)[:, 1]
        return np.where(probs >= threshold, 1, 0)

    def save(self, filepath):
        """Saves trained scratch model to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """Loads scratch model from pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    # Simple test run on dummy data
    X_test = np.random.randn(100, 5)
    y_test = np.random.randint(0, 2, 100)
    model = ScratchLogisticRegression(learning_rate=0.1, n_iterations=200)
    model.fit(X_test, y_test)
    preds = model.predict(X_test)
    print("Scratch Logistic Regression unit test passed! Predictions shape:", preds.shape)
