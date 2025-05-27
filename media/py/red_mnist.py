# red_mnist.py — NumPy-only neural network for MNIST classification

import numpy as np
import pandas as pd
from pathlib import Path

def load_mnist_csv(csv_path: str):
    """Load and normalize MNIST data from a Kaggle CSV."""
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(np.float32) / 255.0
    return features.T, labels  # (784, m), (m,)

def one_hot(labels: np.ndarray, num_classes: int = 10):
    """Convert integer labels to one-hot encoded matrix."""
    m = labels.size
    oh = np.zeros((num_classes, m))
    oh[labels, np.arange(m)] = 1
    return oh  # (10, m)

def one_ice(numbers: np.ndarray):
    """Return from one-hot enconded to dec labels"""
    return np.argmax(numbers, axis=0)

def sigmoid(z):
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid, given activation a."""
    return a * (1 - a)

def softmax(z):
    """Numerically stable softmax."""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def cross_entropy_loss(Y_pred: np.ndarray, Y_true: np.ndarray):
    """Categorical cross-entropy loss."""
    m = Y_true.shape[1]
    eps = 1e-9
    return -(1/m) * np.sum(Y_true * np.log(Y_pred + eps))

class MNISTNetwork:
    """Two-layer fully connected neural network for MNIST."""

    def __init__(self,
                 input_size: int = 784,
                 hidden_size: int = 128,
                 output_size: int = 10,
                 learning_rate: float = 0.1):
        self.lr = learning_rate
        # Xavier/Glorot initialization for sigmoid layers
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(1/input_size)
        self.b1 = np.random.randn(hidden_size, 1) * 0.01
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(1/hidden_size)
        self.b2 = np.random.randn(output_size, 1) * 0.01

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute forward pass."""
        self.Z1 = self.W1 @ X + self.b1       # (hidden_size, m)
        self.A1 = sigmoid(self.Z1)            # (hidden_size, m)
        self.Z2 = self.W2 @ self.A1 + self.b2 # (output_size, m)
        self.A2 = softmax(self.Z2)            # (output_size, m)
        return self.A2

    def backward(self, X: np.ndarray, Y: np.ndarray):
        """Compute gradients and update weights/biases."""
        m = X.shape[1]
        # Output layer gradients
        dZ2 = self.A2 - Y                     # (output_size, m)
        dW2 = (1/m) * dZ2 @ self.A1.T         # (output_size, hidden_size)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # (output_size,1)
        # Hidden layer gradients
        dA1 = self.W2.T @ dZ2                 # (hidden_size, m)
        dZ1 = dA1 * sigmoid_derivative(self.A1)          # (hidden_size, m)
        dW1 = (1/m) * dZ1 @ X.T               # (hidden_size, input_size)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True) # (hidden_size,1)
        # Parameter updates
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    @staticmethod
    def accuracy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """Compute classification accuracy."""
        pred_labels = np.argmax(Y_pred, axis=0)
        true_labels = np.argmax(Y_true, axis=0)
        return np.mean(pred_labels == true_labels)

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              epochs: int = 20,
              batch_size: int = 64):
        """Train network with mini-batch SGD."""
        m = X.shape[1]
        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(m)
            X_shuf = X[:, perm]
            Y_shuf = Y[:, perm]
            for i in range(0, m, batch_size):
                X_batch = X_shuf[:, i:i+batch_size]
                Y_batch = Y_shuf[:, i:i+batch_size]
                self.forward(X_batch)
                self.backward(X_batch, Y_batch)
            Y_pred = self.forward(X)
            loss = cross_entropy_loss(Y_pred, Y)
            acc = MNISTNetwork.accuracy(Y_pred, Y)
            print(f"Epoch {epoch:02d} | loss {loss:.4f} | accuracy {acc:.2%}")

if __name__ == "__main__":
    csv_path = Path("../csv/numbers.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Missing train.csv — download it from Kaggle and place it here.")
    X, y = load_mnist_csv(str(csv_path))    # X shape (784, m), y shape (m,)
    Y = one_hot(y)                          # Y shape (10, m)
    nn = MNISTNetwork(learning_rate=0.1)
    reforce = True
    weights = None
    try:
        weights = np.load("mnist_weights.npz")
        nn.W1 = weights["W1"]
        nn.b1 = weights["b1"]
        nn.W2 = weights["W2"]
        nn.b2 = weights["b2"]
        print("Loaded pre-trained weights from mnist_weights.npz")
    except FileNotFoundError:
        print("No pre-trained weights found. Training from scratch.")
    if (not weights) or reforce:
        nn.train(X, Y, epochs=500, batch_size=128)
        # Save trained parameters
        np.savez("mnist_weights.npz", W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)
        print("\nTraining complete. Weights saved to mnist_weights.npz")
    Y_pred = nn.forward(X)
    numbers_pred = one_ice(Y_pred)
    numbers_true = one_ice(Y)
    """ df with origin, predicted """
    headers = ["origin", "predicted"]
    df = pd.DataFrame(columns=headers)
    df["origin"] = numbers_true
    df["predicted"] = numbers_pred
    df = df[df["origin"] != df["predicted"]]
    print(df.head(10))
    
