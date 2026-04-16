import numpy as np
import matplotlib.pyplot as plt


class ManualRegressor:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """The Optimization Loop (The 'Learning' Phase)"""
        n_samples, n_features = X.shape

        # 1. Initialize parameters (Zero initialization)
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            # 2. Forward Pass: Prediction (Linear Transformation)
            y_predicted = np.dot(X, self.weights) + self.bias

            # 3. Calculate Gradients (The Calculus Step)
            # error = (y_actual - y_predicted)
            error = y - y_predicted

            # dw = (-2/n) * X^T * error
            dw = (-2 / n_samples) * np.dot(X.T, error)
            # db = (-2/n) * sum(error)
            db = (-2 / n_samples) * np.sum(error)

            # 4. Update Rule (Gradient Descent)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Optional: Track Loss to ensure convergence
            loss = np.mean(np.square(error))
            self.loss_history.append(loss)

    def predict(self, X):
        """Inference: Using the learned weights on new data"""
        return np.dot(X, self.weights) + self.bias


# --- MAIN EXECUTION ---

# 1. Generate Synthetic Dataset (y = 2x + 5 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # 100 points between 0 and 2
y = 5 + 2 * X.flatten() + np.random.randn(100) * 0.5  # True slope=2, intercept=5

# 2. Train the Model
model = ManualRegressor(learning_rate=0.1, iterations=100)
model.fit(X, y)

# 3. Predict for plotting
predictions = model.predict(X)

print(f"Learned Weight: {model.weights[0]:.4f} (Goal: 2.0)")
print(f"Learned Bias: {model.bias:.4f} (Goal: 5.0)")

# 4. Plot the Results using Matplotlib
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, predictions, color='red', linewidth=2, label='AI Learning (Regression Line)')
plt.xlabel("X (Input)")
plt.ylabel("y (Target)")
plt.title("Linear Regression from Scratch (NumPy)")
plt.legend()
plt.show()