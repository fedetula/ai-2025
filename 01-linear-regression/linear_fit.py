import numpy as np
import matplotlib.pyplot as plt

# Define data
DATASIZE_LEN = 20
x = np.random.rand(DATASIZE_LEN)
noise1 = np.random.normal(0, 0.1, DATASIZE_LEN)
y = 3.2 * x + noise1


# Generic split function for multiple iterables
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


# Split data into training and test sets
x_train, y_train, x_test, y_test = split(x, y, percentage=0.8)

print(x_train, x_test)
print(y_train, y_test)

X_train = np.vstack([np.ones_like(x_train), x_train]).T  # Transpose to shape (N, 2)


def LinearRegression(X, w):
    return X @ w


def MSE_Loss(w, model, X, y):
    return np.mean((model(X, w) - y) ** 2)


# Stochastic gradient descent optimizer
def SGD(w, model, X, y, lr=0.01):
    return w - lr * 2 * X.T @ (model(X, w) - y)  # Gradient computation


def train(X, y, model, loss, optimizer, epochs=1000):
    w = np.random.rand(X.shape[1])  # w has to have 2 elements (shape (2,))
    for _ in range(epochs):
        w = optimizer(w, model, X, y)
        print(loss(w, model, X, y))
    return w


w = train(X_train, y_train, LinearRegression, MSE_Loss, SGD)

print("Learned weights:", w)

plt.scatter(x_train, y_train, label="Data points")
plt.plot(
    x_train,
    LinearRegression(np.vstack([np.ones_like(x_train), x_train]).T, w),
    color="red",
    label="Fitted line",
)
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
