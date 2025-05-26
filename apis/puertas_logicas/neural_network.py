import numpy as np

# ---------- Neural network class ----------

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, learning_rate: float = 0.1, save_weights: str = "", activation_function: str = "sig"):
        self.lr = learning_rate
        # Weights and biases for hidden layer
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.random.randn(hidden_size, 1) * 0.01
        # Weights and biases for output layer
        self.W2 = np.random.randn(1, hidden_size) * 0.01
        self.b2 = np.random.randn(1, 1) * 0.01
        self.save_weights = save_weights
        self.last_loss_str = ""
        self.activation_function = activation_function
    
    def sigmoid(self, z):
        if self.activation_function == "sig":
            return 1 / (1 + np.exp(-z))
        elif self.activation_function == "tanh":
            return np.tanh(z)

    def sigmoid_derivative(self, a):
        if self.activation_function == "sig":
            return a * (1 - a)
        elif self.activation_function == "tanh":
            return 1 - a**2

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: compute hidden and output activations."""
        self.Z1 = self.W1 @ X + self.b1          # Linear transform hidden
        self.A1 = self.sigmoid(self.Z1)               # Activation hidden
        self.Z2 = self.W2 @ self.A1 + self.b2    # Linear transform output
        self.A2 = self.sigmoid(self.Z2)               # Activation output
        return self.A2

    def compute_loss(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        m = Y.shape[1]
        eps = 1e-8
        loss = -(1 / m) * np.sum(
            Y * np.log(Y_pred + eps) +
            (1 - Y) * np.log(1 - Y_pred + eps)
        )
        return loss

    def backpropagate(self, X: np.ndarray, Y: np.ndarray):
        """Backward pass: compute gradients and update parameters."""
        m = X.shape[1]
        # Output layer error
        dZ2 = self.A2 - Y
        dW2 = (1 / m) * dZ2 @ self.A1.T
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        # Hidden layer error
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = (1 / m) * dZ1 @ X.T
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        # Parameter update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X: np.ndarray, Y: np.ndarray,
              epochs: int = 10000, print_every: int = 1000):
        """Training loop."""
        for epoch in range(1, epochs + 1):
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y_pred, Y)
            self.backpropagate(X, Y)
            if epoch % print_every == 0:
                print(f"Epoch {epoch}, loss: {loss:.6f}")
        last_Y_pred = self.forward(X)
        last_loss = self.compute_loss(last_Y_pred, Y)
        print(f"Final loss: {last_loss:.6f}, {last_Y_pred}")
        self.last_loss_str = f"{last_loss:.6f}"
        if self.save_weights:
            np.savez(self.save_weights, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_weights(self, filename):
        with np.load(filename) as data:
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
    
    def predict(self, X):
        return self.forward(X)

