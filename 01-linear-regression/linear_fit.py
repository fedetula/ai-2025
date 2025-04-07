import numpy as np
import matplotlib.pyplot as plt


# Linear Regression Model as a Class
class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=1000, lr=0.01):
        # Initialize weights (including bias) to match the number of features + 1 (for the bias term)
        self.w = np.random.rand(self._bias(X).shape[1])

        # Gradient Descent for training
        for _ in range(epochs):
            self.w = self._sgd(X, y, lr)

    def _sgd(self, X, y, lr):
        return self.w - lr * 2 * self._bias(X).T @ (self.predict(X) - y)

    def _mse_loss(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)

    def _bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def predict(self, X):
        return self._bias(X) @ self.w


# Define data
DATASIZE_LEN = 20
x1 = np.random.rand(DATASIZE_LEN)  # First feature
x2 = np.random.rand(DATASIZE_LEN)  # Second feature
x3 = np.random.rand(DATASIZE_LEN)  # Third feature

# Combine the features into a matrix X (each row is a data point, each column is a feature)
X = np.vstack([x1, x2, x3]).T  # Shape (20, 3) -- 20 samples, 3 features

# Target y is a linear combination of features with some noise
w_true = np.array([3.2, 1.9, 2.5])  # True coefficients for the feature
noise = np.random.normal(0, 0.1, DATASIZE_LEN)
bias = -1.4
y = X @ w_true + bias + noise


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

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train, epochs=1000, lr=0.01)

print("Learned weights:", model.w)

y_pred = model.predict(X_test)

# Plotting the data points and the fitted line
plt.scatter(
    X_train[:, 0], y_train, label="Data X1"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 0],
    model.predict(X_train),
    color="red",
    label="Predicted-X1",
)
plt.scatter(
    X_train[:, 1], y_train, label="Data X2"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 1],
    model.predict(X_train),
    color="green",
    label="Predicted-X2",
)
plt.scatter(
    X_train[:, 2], y_train, label="Data X3"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 2],
    model.predict(X_train),
    color="blue",
    label="Predicted-X3",
)
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
