# red_mnist.py — Red neuronal «desde cero» (solo NumPy) para clasificar MNIST
# ---------------------------------------------------------------------------------
# Este script implementa una red de una capa oculta (784‑128‑10) y ENTRENAMIENTO
# con descenso de gradiente estocástico por mini‑lotes.
# Todas las funciones y variables están en español y se comentan con detalle para
# que puedas seguir el flujo de datos paso a paso.
# ---------------------------------------------------------------------------------
# Requisitos previos:
#   1. Descarga `train.csv` del concurso “Digit Recognizer” de Kaggle y colócalo
#      junto a este archivo. Cada fila: etiqueta (0‑9) + 784 píxeles (0‑255).
#   2. Solo se usan NumPy y pandas (no TensorFlow / PyTorch). Instálalos con:
#         pip install numpy pandas
# ---------------------------------------------------------------------------------

# =======================
# IMPORTACIONES BÁSICAS
# =======================
import numpy as np            # Álgebra lineal y arrays de alta velocidad
import pandas as pd           # Lectura de CSV y manipulación tabular
from pathlib import Path      # Manejo de rutas multiplataforma

# ==================================
# FUNCIONES AUXILIARES DE CARGA I/O
# ==================================

def cargar_mnist_csv(ruta_csv: str):
    """Lee el CSV de Kaggle y produce los datos normalizados.

    Args:
        ruta_csv: Ruta al archivo CSV (p. ej. "train.csv").

    Returns:
        X: Matriz de entrada de forma (784, m) con valores en [0,1].
        y: Vector de etiquetas de forma (m,) con enteros 0‑9.
    """
    df = pd.read_csv(ruta_csv)                  # Cargar todo el CSV en un DataFrame
    y = df.iloc[:, 0].values                   # Primera columna → etiquetas
    X = df.iloc[:, 1:].values.astype(np.float32) / 255.0  # Resto → píxeles normalizados
    return X.T, y                              # Transponer para obtener (características, muestras)


def one_hot(etiquetas: np.ndarray, clases: int = 10):
    """Convierte un vector de números (0‑9) en una matriz one‑hot.

    Ej.: y=[3,0]  →  [[0,1], [0,0], [0,0], [1,0], ..., [0,0]]
    """
    m = etiquetas.size                         # Número de ejemplos
    oh = np.zeros((clases, m))                 # Matriz llena de ceros (10 × m)
    oh[etiquetas, np.arange(m)] = 1            # Colocar 1 en la fila de la etiqueta
    return oh

# ============================
# FUNCIONES DE ACTIVACIÓN
# ============================

def sigmoide(z):
    """Sigmoide clásica: aplana valores al rango (0,1)."""
    return 1 / (1 + np.exp(-z))


def derivada_sigmoide(a):
    """Derivada de la sigmoide expresada en función de la activación ya calculada."""
    return a * (1 - a)


# ---------- Softmax + pérdida ----------

def softmax(z):
    """Convierte logits en distribuciones de probabilidad estables numéricamente."""
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Restar máx evita overflow
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def perdida_cross_entropy(Y_pred, Y_true):
    """Entropía cruzada categórica promedio (E_pred contra Y_true)."""
    m = Y_true.shape[1]
    eps = 1e-9                                           # Pequeño valor para evitar log(0)
    return -(1 / m) * np.sum(Y_true * np.log(Y_pred + eps))

# ============================
# DEFINICIÓN DE LA RED
# ============================
class RedMNIST:
    """Red neuronal con una sola capa oculta totalmente conectada."""

    def __init__(self, entradas: int = 784, ocultas: int = 128, salidas: int = 10, tasa: float = 0.1):
        # ---------------- PARAMETROS DE USUARIO ----------------
        self.tasa = tasa                                 # Tasa de aprendizaje η

        # ---------------- PESOS Y SESGOS ----------------
        # Inicialización Xavier/Glorot para capas con sigmoide
        self.W1 = np.random.randn(ocultas, entradas) * np.sqrt(1 / entradas)
        self.b1 = np.random.randn(ocultas, 1) * 0.01     # Sesgos pequeños ≠0 para romper simetría
        self.W2 = np.random.randn(salidas, ocultas) * np.sqrt(1 / ocultas)
        self.b2 = np.random.randn(salidas, 1) * 0.01

    # ------------ FORWARD PASS ------------
    def adelantado(self, X: np.ndarray):
        """Computa la salida de la red para un batch de entrada X."""
        # Capa oculta: Z1 = W1·X + b1  →  A1 = sigmoide(Z1)
        self.Z1 = self.W1 @ X + self.b1
        self.A1 = sigmoide(self.Z1)
        # Capa salida: Z2 = W2·A1 + b2  →  A2 = softmax(Z2)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    # ------------ BACKWARD PASS ------------
    def retropropagacion(self, X: np.ndarray, Y: np.ndarray):
        """Calcula gradientes y actualiza parámetros usando descenso de gradiente."""
        m = X.shape[1]
        # Error capa salida: dZ2 = A2 - Y  (para softmax + CE)
        dZ2 = self.A2 - Y                                   # (10, m)
        # Gradientes capa salida
        dW2 = (1 / m) * dZ2 @ self.A1.T                    # (10,128)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True) # (10,1)

        # Propagar al nivel anterior
        dA1 = self.W2.T @ dZ2                              # (128,m)
        dZ1 = dA1 * derivada_sigmoide(self.A1)             # (128,m)
        dW1 = (1 / m) * dZ1 @ X.T                          # (128,784)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True) # (128,1)

        # Actualización de parámetros
        self.W2 -= self.tasa * dW2
        self.b2 -= self.tasa * db2
        self.W1 -= self.tasa * dW1
        self.b1 -= self.tasa * db1

    # ------------ ENTRENAMIENTO ------------
    def entrenar(self, X: np.ndarray, Y: np.ndarray, epocas: int = 20, tam_lote: int = 64):
        """Entrena la red con mini‑batch SGD y reporta métricas por época."""
        m = X.shape[1]
        for epoch in range(1, epocas + 1):
            # 1) Mezclar índices para aleatoriedad cada época
            indices = np.random.permutation(m)
            X_barajado, Y_barajado = X[:, indices], Y[:, indices]

            # 2) Recorrer datos en mini‑lotes
            for i in range(0, m, tam_lote):
                X_batch = X_barajado[:, i:i + tam_lote]
                Y_batch = Y_barajado[:, i:i + tam_lote]
                self.adelantado(X_batch)       # Forward
                self.retropropagacion(X_batch, Y_batch)  # Backward + update

            # 3) Métricas de la época completa
            Y_pred = self.adelantado(X)
            loss = perdida_cross_entropy(Y_pred, Y)
            acc = self.precision(Y_pred, Y)
            print(f"Época {epoch:02d} | pérdida {loss:.4f} | precisión {acc:.2%}")

    # ---------------- MÉTRICA AUXILIAR ----------------
    @staticmethod
    def precision(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """Calcula la fracción de aciertos entre la clase predicha y la verdadera."""
        pred_clases = np.argmax(Y_pred, axis=0)
        true_clases = np.argmax(Y_true, axis=0)
        return np.mean(pred_clases == true_clases)

# ==================================
# BLOQUE PRINCIPAL DE EJECUCIÓN
# ==================================
if __name__ == "__main__":
    # 1. Verificar que el CSV exista
    ruta = Path("train.csv")
    if not ruta.exists():
        raise FileNotFoundError("Falta train.csv — descárgalo de Kaggle y colócalo aquí.")

    # 2. Cargar y preparar datos
    X, y = cargar_mnist_csv(str(ruta))         # X (784, m), y (m,)
    Y_onehot = one_hot(y)                      # Y_onehot (10, m)

    # 3. Instanciar y entrenar la red
    red = RedMNIST(tasa=0.5)                   # tasa=0.5 algo agresiva pero funciona
    red.entrenar(X, Y_onehot, epocas=100, tam_lote=128)

    # 4. Guardar parámetros entrenados para uso futuro (numpy .npz)
    np.savez("mnist_pesos.npz", W1=red.W1, b1=red.b1, W2=red.W2, b2=red.b2)

    print("\nEntrenamiento completo. Pesos guardados en mnist_pesos.npz")
