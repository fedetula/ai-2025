import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a simple binary classification dataset
X, y = make_classification(
    n_samples=100,  # Number of samples
    n_features=2,  # Number of features
    n_informative=2,  # Number of informative features
    n_redundant=0,  # No redundant features
    n_classes=2,  # Binary classification (2 classes)
    random_state=42,  # Set random state for reproducibility
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the neural network using MLPClassifier
# We'll use a simple neural network with 1 hidden layer containing 5 units
mlp = MLPClassifier(
    hidden_layer_sizes=(5,), activation="logistic", max_iter=1000, random_state=42
)
mlp.fit(X_train, y_train)

# Print the accuracy of the model on the test set
print(f"Test set accuracy: {mlp.score(X_test, y_test):.2f}")

# Plotting the decision boundary and the data points
xx, yy = np.meshgrid(
    np.linspace(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1, 100),
    np.linspace(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 100),
)

# Predict labels for each point in the meshgrid
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
plt.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, edgecolors="k", marker="o", cmap="coolwarm"
)
plt.title("Neural Network Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
