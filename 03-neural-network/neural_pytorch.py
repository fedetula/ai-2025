import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate a simple binary classification dataset
X, y = make_classification(
    n_samples=100,  # Number of samples
    n_features=2,  # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,  # No redundant features
    n_classes=2,  # Binary classification (2 classes)
    random_state=42,  # Set random state for reproducibility
)

# Standardize the features to have mean=0 and variance=1 (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert the data into PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Make y a column vector

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42
)


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define a simple 2-layer neural network
        self.fc1 = nn.Linear(2, 5)  # 2 input features, 5 hidden units
        self.fc2 = nn.Linear(5, 1)  # 5 hidden units, 1 output unit
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)  # ReLU activation for hidden layer
        x = self.fc2(x)
        x = self.sigmoid(x)  # Sigmoid for output layer to get probabilities
        return x


# Instantiate the model
model = NeuralNetwork()

# Define the loss function (Binary Cross-Entropy Loss)
criterion = nn.BCELoss()

# Define the optimizer (Stochastic Gradient Descent)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the model
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Test the model
with torch.no_grad():
    predicted = model(X_test)
    predicted = predicted.round()  # Round to get binary predictions
    accuracy = (predicted.eq(y_test).sum().item()) / y_test.size(0)
    print(f"Test accuracy: {accuracy:.2f}")

# Plotting the decision boundary and the data points
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100),
    np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 100),
)

Z = (
    model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    .detach()
    .numpy()
    .reshape(xx.shape)
)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c=y_train.squeeze(),
    edgecolors="k",
    marker="o",
    cmap="coolwarm",
)
plt.title("Neural Network Decision Boundary (PyTorch)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
