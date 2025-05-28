# red_mnist.py — NumPy-only neural network for MNIST classification

import numpy as np
import pandas as pd
from pathlib import Path
from rn import RedNeuronal

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

if __name__ == "__main__":
    csv_path = Path("../csv/numbers.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Missing train.csv — download it from Kaggle and place it here.")
    X, y = load_mnist_csv(str(csv_path))    # X shape (784, m), y shape (m,)
    Y = one_hot(y)                          # Y shape (10, m)
    
    capas = [784, 128, 10]
    activations = ["relu", "sigmoid"]
    learning_rate = 0.1
    
    nn = RedNeuronal(capas, activations=activations, lr=learning_rate)
    nn.fit(X, Y, epochs=5000, batch_size=32, verbose=100)
    nn.save("pesos.json")
    
""" 

"""