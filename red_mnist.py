# red_mnist.py — Red neuronal «desde cero» (solo NumPy) para clasificar MNIST
# Variables y comentarios en español, con impresiones mínimas y claras.
# Descarga previa necesaria: dataset MNIST en CSV (train.csv, test.csv) o formato IDX.
#  - train.csv (Kaggle) → primeras 785 columnas: label, pixel0…pixel783
#  - test.csv (solo píxeles)
# Cambio simple: coloca los CSV en la misma carpeta del script.

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- Utilidades ----------------

def cargar_mnist_csv(ruta_csv: str):
    """Lee un CSV de Kaggle y devuelve X ∈ [0,1] y y como enteros."""
    df = pd.read_csv(ruta_csv)
    y = df.iloc[:, 0].values  # primera columna: etiqueta
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0  # normalizar píxeles
    return X.T, y  # X shape (784, m), y shape (m,)


def one_hot(etiquetas: np.ndarray, clases: int = 10):
    """Convierte un vector de etiquetas (m,) en matriz one‑hot (clases, m)."""
    m = etiquetas.size
    oh = np.zeros((clases, m))
    oh[etiquetas, np.arange(m)] = 1
    return oh

# ---------------- Activaciones ----------------

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def derivada_sigmoide(a):
    return a * (1 - a)

# Softmax + pérdida cross‑entropy en salida

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def perdida_cross_entropy(Y_pred, Y_true):
    m = Y_true.shape[1]
    eps = 1e-9
    return -(1 / m) * np.sum(Y_true * np.log(Y_pred + eps))

# ---------------- Red neuronal ----------------

class RedMNIST:
    def __init__(self, entradas: int = 784, ocultas: int = 128, salidas: int = 10, tasa: float = 0.1):
        self.tasa = tasa
        # Xavier para sigmoide: √(1/n)
        self.W1 = np.random.randn(ocultas, entradas) * np.sqrt(1 / entradas)
        self.b1 = np.random.randn(ocultas, 1) * 0.01
        self.W2 = np.random.randn(salidas, ocultas) * np.sqrt(1 / ocultas)
        self.b2 = np.random.randn(salidas, 1) * 0.01

    def adelantado(self, X):
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = sigmoide(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def retropropagacion(self, X, Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y  # (10, m)
        dW2 = (1 / m) * dZ2 @ self.A1.T
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * derivada_sigmoide(self.A1)
        dW1 = (1 / m) * dZ1 @ X.T
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Actualizar
        self.W2 -= self.tasa * dW2
        self.b2 -= self.tasa * db2
        self.W1 -= self.tasa * dW1
        self.b1 -= self.tasa * db1

    def entrenar(self, X, Y, epocas=20, tam_lote=64):
        m = X.shape[1]
        for epoch in range(1, epocas + 1):
            # Mezclar datos
            indices = np.random.permutation(m)
            X_barajado = X[:, indices]
            Y_barajado = Y[:, indices]
            # Mini‑batch SGD
            for i in range(0, m, tam_lote):
                X_batch = X_barajado[:, i:i+tam_lote]
                Y_batch = Y_barajado[:, i:i+tam_lote]
                self.adelantado(X_batch)
                self.retropropagacion(X_batch, Y_batch)
            # Evaluación cada época
            Y_pred = self.adelantado(X)
            loss = perdida_cross_entropy(Y_pred, Y)
            acc = self.precision(Y_pred, Y)
            print(f"Época {epoch:02d} | pérdida {loss:.4f} | precisión {acc:.2%}")

    @staticmethod
    def precision(Y_pred, Y_true):
        pred_clases = np.argmax(Y_pred, axis=0)
        true_clases = np.argmax(Y_true, axis=0)
        return np.mean(pred_clases == true_clases)

# ---------------- Ejecución ----------------

if __name__ == "__main__":
    ruta = Path("mnist_test.csv")
    if not ruta.exists():
        raise FileNotFoundError("Descarga train.csv de Kaggle MNIST y colócalo junto al script.")

    X, y = cargar_mnist_csv(str(ruta))
    Y_onehot = one_hot(y)

    red = RedMNIST(tasa=0.5)
    red.entrenar(X, Y_onehot, epocas=15, tam_lote=128)

    # Guardar parámetros entrenados (opcional)
    np.savez("mnist_pesos.npz", W1=red.W1, b1=red.b1, W2=red.W2, b2=red.b2)
