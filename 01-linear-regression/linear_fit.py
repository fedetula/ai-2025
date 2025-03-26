import numpy as np
import matplotlib.pyplot as plt

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
#noise = 0
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

# Prepare the design matrix for training (adding bias term)
X_train_bias = np.hstack(
    [np.ones((X_train.shape[0], 1)), X_train]
) 


def LinearRegression(X, w):
    return X @ w


def MSE_Loss(w, model, X, y):
    return np.mean((model(X, w) - y) ** 2)


def SGD(w, model, X, y, lr=0.01):
    return w - lr * 2 * X.T @ (model(X, w) - y)  # Gradient computation


# Training function
def train(X, y, model, loss, optimizer, epochs=1000):
    w = np.random.rand(
        X.shape[1]
    ) 
    for _ in range(epochs):
        w = optimizer(w, model, X, y)
        print(loss(w, model, X, y))  # Print loss for debugging
    return w


w = train(X_train_bias, y_train, LinearRegression, MSE_Loss, SGD)

print("Learned weights:", w)

# Plotting the data points and the fitted line
plt.scatter(
    X_train[:, 0], y_train, label="Data X1"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 0],
    LinearRegression(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), w),
    color="red",
    label="Predicted-X1",
)
plt.scatter(
    X_train[:, 1], y_train, label="Data X2"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 1],
    LinearRegression(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), w),
    color="green",
    label="Predicted-X2",
)
plt.scatter(
    X_train[:, 2], y_train, label="Data X3"
)  # Use the first feature for plotting
plt.scatter(
    X_train[:, 2],
    LinearRegression(np.hstack([np.ones((X_train.shape[0], 1)), X_train]), w),
    color="blue",
    label="Predicted-X3",
)
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
