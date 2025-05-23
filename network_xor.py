# network_xor.py
# Implementación detallada de una red neuronal desde cero para resolver la compuerta XOR

import numpy as np  # Biblioteca fundamental para operaciones numéricas y manejo de arrays multidimensionales

# 1. Función de activación sigmoide y su derivada
#    La sigmoide introduce no linealidad y mapea valores reales al rango (0,1).
def sigmoid(z):
    # z: valor o array de valores de entrada (suma ponderada más sesgo)
    # Devuelve: 1 / (1 + e^{-z})
    return 1 / (1 + np.exp(-z))

# Derivada de la sigmoide necesaria para backpropagation (regla de la cadena)
def sigmoid_derivative(a):
    # a: valor de activación calculado por sigmoid(z)
    # Derivada: sigmoid(z) * (1 - sigmoid(z))
    return a * (1 - a)

class NeuralNetwork:
    """
    Clase que implementa una red neuronal de 2 capas (oculta + salida) para resolver XOR.
    Conceptos clave:
      - Pesos (W) y sesgos (b) como parámetros entrenables
      - Forward pass y backpropagation
      - Descenso de gradiente para actualización de parámetros
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # input_size: número de neuronas de la capa de entrada (para XOR, 2)
        # hidden_size: número de neuronas en la capa oculta (para capacidad de aprender no linealidad)
        # output_size: neuronas de salida (para XOR, 1)
        # learning_rate: tasa de aprendizaje (hiperparámetro para controlar tamaño de paso en gradiente)
        self.lr = learning_rate

        # Inicialización de los pesos con valores pequeños aleatorios (para romper simetría)
        # W1: matriz de pesos de capa oculta a entrada (hidden_size x input_size)
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        # b1: vector de sesgos para capa oculta (hidden_size x 1)
        self.b1 = np.zeros((hidden_size, 1))

        # W2: matriz de pesos de capa de salida a capa oculta (output_size x hidden_size)
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        # b2: vector de sesgos para capa de salida (output_size x 1)
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        """
        Propagación hacia adelante (forward pass): calcula las activaciones de cada capa.
        X: array de entrada con forma (input_size, m) donde m es el número de ejemplos.
        Retorna A2: activación de la capa de salida (predicciones) con forma (output_size, m).
        """
        # Cálculo de suma ponderada + sesgo en capa oculta: Z1 = W1·X + b1
        self.Z1 = np.dot(self.W1, X) + self.b1
        # Aplicar función de activación sigmoide: A1 = sigmoid(Z1)
        self.A1 = sigmoid(self.Z1)

        # Segunda capa: suma ponderada + sesgo: Z2 = W2·A1 + b2
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        # Activación de salida con sigmoide: A2 = sigmoid(Z2)
        self.A2 = sigmoid(self.Z2)

        # A2 son las predicciones de la red para cada ejemplo
        return self.A2

    def compute_loss(self, Y_hat, Y):
        """
        Calcula la pérdida (loss) usando entropía cruzada binaria.
        Y_hat: predicciones (A2)
        Y: etiquetas reales (0 o 1)
        Retorna valor escalar de pérdida media.
        """
        m = Y.shape[1]  # número de ejemplos
        epsilon = 1e-8  # evita log(0) y división por cero
        # Fórmula de entropía cruzada: -(1/m) * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        loss = - (1/m) * np.sum(Y * np.log(Y_hat + epsilon) + (1 - Y) * np.log(1 - Y_hat + epsilon))
        return loss

    def backward(self, X, Y):
        """
        Propagación hacia atrás (backpropagation): calcula gradientes y actualiza parámetros.
        """
        m = X.shape[1]  # número de ejemplos

        # 1) Gradiente de la capa de salida
        # dZ2: derivada de la pérdida respecto a Z2 (para entropía + sigmoide)
        dZ2 = self.A2 - Y  # derivada directa para entropía cruzada binaria
        # dW2: gradiente de W2: (1/m) * dZ2·A1^T
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        # db2: gradiente de b2: media de dZ2 sobre ejemplos
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        # 2) Propagar error a capa oculta
        # dA1: derivada de la pérdida respecto a A1
        dA1 = np.dot(self.W2.T, dZ2)
        # dZ1: dA1 * derivada de la función de activación sigmoide
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        # dW1: gradiente de W1
        dW1 = (1/m) * np.dot(dZ1, X.T)
        # db1: gradiente de b1
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        # 3) Actualizar parámetros con descenso de gradiente
        # W := W - lr * dW ; b := b - lr * db
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=10000, print_every=1000):
        """
        Entrena la red neuronal.
        epochs: número de iteraciones sobre todo el dataset.
        print_every: cada cuántas épocas mostrar la pérdida.
        """
        for i in range(1, epochs + 1):
            # Forward pass: obtener predicciones
            Y_hat = self.forward(X)
            # Calcular pérdida
            loss = self.compute_loss(Y_hat, Y)
            # Backward pass: actualizar pesos y sesgos
            self.backward(X, Y)

            # Mostrar progreso cada "print_every" iteraciones
            if i % print_every == 0:
                print(f"Época {i}, loss: {loss:.6f}")  # .6f: formato con 6 decimales

if __name__ == "__main__":
    # ===== Datos de entrenamiento para XOR =====
    # X: entradas con forma (2,4): 2 características, 4 ejemplos
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    # Y: etiquetas con forma (1,4): resultado XOR para cada ejemplo
    Y = np.array([[0, 1, 1, 0]])

    # Crear la red neuronal
    # Parámetros: 2 entradas, 2 neuronas ocultas, 1 salida, lr=1 (experimentar con lr)
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=1)

    # Entrenar la red
    nn.train(X, Y, epochs=10000, print_every=1000)

    # Probar la red con los mismos datos de entrada
    outputs = nn.forward(X)
    # Mostrar resultados aproximados (round a 3 decimales)
    print("\nPredicciones finales (aproximadas):")
    print(np.round(outputs, 3))

# Temas para profundizar:
# - Álgebra lineal: multiplicación de matrices, vectores y dimensiones
# - Funciones de activación: por qué se usan (no linealidad)
# - Descenso de gradiente y tasa de aprendizaje (learning rate)
# - Backpropagation: regla de la cadena y cálculo de derivadas parciales
