import numpy as np
import matplotlib.pyplot as plt


# Logistic Regression Model as a Class
class LogisticRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=1000, lr=0.01):
        # Initialize weights (including bias) to match the number of features + 1 (for the bias term)
        self.w = np.random.rand(self._bias(X).shape[1])

        # Gradient Descent for training
        for _ in range(epochs):
            self.w = self._sgd(X, y, lr)

    def _sgd(self, X, y, lr):
        return self.w - lr * self._bias(X).T @ (self.predict(X) - y)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict(self, X):
        return self._sigmoid(self._bias(X) @ self.w)


# Define data for binary classification (example)
DATASIZE_LEN = 100
X = np.random.rand(DATASIZE_LEN, 2)  # Two features for 100 samples

# Let's create a simple linear decision boundary for generating labels
# This will create data points around the decision boundary (e.g., y = 0.5 * x1 + 0.5 * x2 + noise)
y = (X[:, 0] + X[:, 1] + np.random.randn(DATASIZE_LEN) * 0.1 > 1).astype(
    int
)  # Binary labels (0 or 1)


# Split the data into training and testing sets
def split(*iterables, percentage=0.8, shuffle=False):
    if not iterables:
        raise ValueError("At least one iterable is required")

    length = len(iterables[0])
    if any(len(it) != length for it in iterables):
        raise ValueError("All iterables must have the same length")

    if shuffle:
        indices = np.random.permutation(length)
        iterables = [np.array(it)[indices] for it in iterables]

    mid = int(length * percentage)
    split_iterables = [(it[:mid], it[mid:]) for it in iterables]

    return tuple(item for pair in zip(*split_iterables) for item in pair)


X_train, y_train, X_test, y_test = split(X, y, percentage=0.8)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train, epochs=1000, lr=0.01)

print("Learned weights:", model.w)

# Plotting the data points and the decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="bwr", label="Data points")
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

# Decision boundary line: w1 * x1 + w2 * x2 + b = 0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors="black")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()
