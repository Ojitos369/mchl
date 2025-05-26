# red_xor.py — Red neuronal mínima (2 capas) para resolver XOR
# Sólo incluye los prints y comentarios imprescindibles

import numpy as np

# ---------- Funciones de activación ----------

def sigmoide(z):
    # return 1 / (1 + np.exp(-z))
    return np.tanh(z)

def derivada_sigmoide(a):
    # return a * (1 - a)
     return 1 - a**2


# ---------- Clase de la red ----------

class RedNeuronal:
    def __init__(self, entradas: int, ocultas: int, tasa: float = 0.1):
        self.tasa = tasa
        self.pesos1 = np.random.randn(ocultas, entradas) * 0.01
        # self.sesgo1 = np.zeros((ocultas, 1))
        self.sesgo1 = np.random.randn(ocultas, 1) * 0.01
        self.pesos2 = np.random.randn(1, ocultas) * 0.01
        # self.sesgo2 = np.zeros((1, 1))
        self.sesgo2 = np.random.randn(1, 1) * 0.01

    # Propagación hacia adelante
    def adelantado(self, X: np.ndarray) -> np.ndarray:
        self.suma1 = self.pesos1 @ X + self.sesgo1
        self.act1 = sigmoide(self.suma1)
        self.suma2 = self.pesos2 @ self.act1 + self.sesgo2
        self.act2 = sigmoide(self.suma2)
        return self.act2

    # Pérdida (entropía cruzada binaria)
    def perdida(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]; eps = 1e-8
        return -(1 / m) * np.sum(Y * np.log(Y_pred + eps) + (1 - Y) * np.log(1 - Y_pred + eps))

    # Retropropagación y actualización de parámetros
    def retropropagacion(self, X: np.ndarray, Y: np.ndarray):
        m = X.shape[1]
        dS2 = self.act2 - Y
        dPesos2 = (1 / m) * dS2 @ self.act1.T
        dSesgo2 = (1 / m) * np.sum(dS2, axis=1, keepdims=True)
        dS1 = (self.pesos2.T @ dS2) * derivada_sigmoide(self.act1)
        dPesos1 = (1 / m) * dS1 @ X.T
        dSesgo1 = (1 / m) * np.sum(dS1, axis=1, keepdims=True)
        self.pesos2 -= self.tasa * dPesos2
        self.sesgo2 -= self.tasa * dSesgo2
        self.pesos1 -= self.tasa * dPesos1
        self.sesgo1 -= self.tasa * dSesgo1

    # Bucle de entrenamiento
    def entrenar(self, X: np.ndarray, Y: np.ndarray, epocas: int = 10000, mostrar_cada: int = 1000):
        for i in range(1, epocas + 1):
            Y_pred = self.adelantado(X)
            loss = self.perdida(Y_pred, Y)
            self.retropropagacion(X, Y)
            if i % mostrar_cada == 0:
                print(f"Época {i}, pérdida: {loss:.6f}")

# ---------- Ejecución ----------

if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # Entradas
    Y = np.array([[0, 1, 1, 0]])                # Salidas esperadas

    red = RedNeuronal(entradas=2, ocultas=2, tasa=0.5)
    red.entrenar(X, Y, epocas=500000, mostrar_cada=4000)

    print("Predicciones finales:", np.round(red.adelantado(X), 10))
