import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases for two layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights and biases
        self.w1 = np.random.randn(input_size, hidden_size)  # Input to hidden weights
        self.b1 = np.zeros((1, hidden_size))  # Hidden biases
        self.w2 = np.random.randn(hidden_size, output_size)  # Hidden to output weights
        self.b2 = np.zeros((1, output_size))  # Output biases

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward(self, X):
        self.Z1 = X @ self.w1 + self.b1  # Weighted sum for hidden layer
        self.A1 = self.sigmoid(self.Z1)  # Apply sigmoid activation
        self.Z2 = self.A1 @ self.w2 + self.b2  # Weighted sum for output layer
        self.A2 = self.sigmoid(self.Z2)  # Apply sigmoid activation (output)
        return self.A2

    def backward(self, X, y):
        # Compute the gradient of the loss with respect to the weights and biases
        m = X.shape[0]  # Number of samples

        # Output layer gradients
        dZ2 = self.A2 - y  # Error at output layer
        dW2 = self.A1.T @ dZ2 / m  # Weight gradients for output layer
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m  # Bias gradients for output layer

        # Hidden layer gradients
        dZ1 = (dZ2 @ self.w2.T) * self.sigmoid_derivative(
            self.Z1
        )  # Error at hidden layer
        dW1 = X.T @ dZ1 / m  # Weight gradients for hidden layer
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m  # Bias gradients for hidden layer

        # Update weights and biases using gradient descent
        self.w1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * dB1
        self.w2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * dB2

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            self.forward(X)

            # Backward pass (gradient calculation)
            self.backward(X, y)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.compute_loss(y)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    def compute_loss(self, y):
        # Binary cross-entropy loss
        m = y.shape[0]
        epsilon = 1e-15  # to avoid log(0)
        y_pred = np.clip(self.A2, epsilon, 1 - epsilon)  # Prevent log(0)
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m

    def predict(self, X):
        # Return predictions after the forward pass
        return self.forward(X)


# Create synthetic binary classification data
def generate_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    y = (
        (X[:, 0] + X[:, 1] > 1).astype(int).reshape(-1, 1)
    )  # Binary target based on sum of features
    return X, y


# Prepare data
X, y = generate_data()

# Standardize the features to mean=0 and std=1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Instantiate the neural network with 2 input features, 5 hidden units, and 1 output unit
nn = NeuralNetwork(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)

# Train the model
nn.train(X, y, epochs=1000)

# Test the model
y_pred = nn.predict(X)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plotting the decision boundary and the data points
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
)

Z = nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
plt.scatter(
    X[:, 0], X[:, 1], c=y.squeeze(), edgecolors="k", marker="o", cmap="coolwarm"
)
plt.title("Neural Network Decision Boundary (Manual Python)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
