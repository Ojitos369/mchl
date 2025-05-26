# red_xor.py
# Implementación detallada de una red neuronal desde cero (sin frameworks) para resolver la compuerta XOR
# Todas las variables y métodos están en español cuando es viable, para facilitar comprensión.
# Incluye impresiones (prints) exhaustivas en cada etapa para seguir la evolución de los parámetros.
import time
import numpy as np  # Operaciones numéricas y manejo de arrays multidimensionales

# 1. Función de activación sigmoide y su derivada
#    Introduce no linealidad y normaliza la salida al rango (0, 1)

def sigmoide(z):
    """Calcula 1 / (1 + e^{-z})."""
    return 1 / (1 + np.exp(-z))


def derivada_sigmoide(a):
    """Devuelve la derivada de la sigmoide: a * (1 - a)."""
    return a * (1 - a)


class RedNeuronal:
    """
    Red neuronal con:
      - Capa oculta: `ocultas` neuronas
      - Capa de salida: 1 neurona (problema binario)
    Variables clave (en español):
      pesos1, sesgo1 → parámetros de la capa oculta
      pesos2, sesgo2 → parámetros de la capa de salida
      suma1, act1 → pre‑activaciones y activaciones en la capa oculta
      suma2, act2 → pre‑activaciones y activaciones en la capa de salida
    """

    def __init__(self, entradas: int, ocultas: int, salidas: int = 1, tasa_aprendizaje: float = 0.1):
        # time.sleep(1)
        print()
        print("--------------------------------------------------")
        print("--------------------   INIT   --------------------")
        print("--------------------------------------------------")
        print()
        self.tasa = tasa_aprendizaje  # η (eta) en descenso de gradiente
        print("self.tasa")
        print(self.tasa)

        # Inicializar pesos con distribución normal pequeña para romper simetría
        self.pesos1 = np.random.randn(ocultas, entradas) * 0.01  # (ocultas × entradas)
        print("self.pesos1")
        print(self.pesos1)
        self.sesgo1 = np.zeros((ocultas, 1))                      # (ocultas × 1)
        print("self.sesgo1")
        print(self.sesgo1)

        self.pesos2 = np.random.randn(salidas, ocultas) * 0.01   # (1 × ocultas)
        print("self.pesos2")
        print(self.pesos2)
        self.sesgo2 = np.zeros((salidas, 1))                      # (1 × 1)
        print("self.sesgo2")
        print(self.sesgo2)

    # ======== PROPAGACIÓN HACIA ADELANTE ========
    def adelantado(self, X: np.ndarray) -> np.ndarray:
        """Computa la salida de la red y muestra valores intermedios."""
        # Capa oculta: suma ponderada + sesgo → sigmoide
        # time.sleep(1)
        print()
        print("--------------------------------------------------")
        print("-----------------   ADELANTADO   -----------------")
        print("--------------------------------------------------")
        print()
        self.suma1 = np.dot(self.pesos1, X) + self.sesgo1
        print("self.suma1")
        print(self.suma1)
        self.act1 = sigmoide(self.suma1)
        print("self.act1")
        print(self.act1)

        # Capa de salida
        self.suma2 = np.dot(self.pesos2, self.act1) + self.sesgo2
        print("self.suma2")
        print(self.suma2)
        self.act2 = sigmoide(self.suma2)
        print("self.act2")
        print(self.act2)

        # Visualizar
        # print("=== Adelante ===")
        # print("suma1 (pre‑activación oculta):\n", self.suma1)
        # print("act1 (activación oculta):\n", self.act1)
        # print("suma2 (pre‑activación salida):\n", self.suma2)
        # print("act2 (salida / predicción):\n", self.act2)

        return self.act2

    # ======== FUNCIÓN DE PÉRDIDA ========
    def perdida(self, Y_pred: np.ndarray, Y: np.ndarray) -> float:
        """Calcula entropía cruzada binaria y la imprime."""
        # time.sleep(1)
        print()
        print("---------------------------------------------------")
        print("-------------------   PERDIDA   -------------------")
        print("---------------------------------------------------")
        print()
        m = Y.shape[1]
        eps = 1e-8  # evitar log(0)
        perdida = -(1 / m) * np.sum(Y * np.log(Y_pred + eps) + (1 - Y) * np.log(1 - Y_pred + eps))
        print("Pérdida (entropía cruzada):", perdida)
        return perdida

    # ======== PROPAGACIÓN HACIA ATRÁS ========
    def retropropagacion(self, X: np.ndarray, Y: np.ndarray):
        """Calcula gradientes y actualiza pesos/sesgos; imprime cada gradiente."""
        # time.sleep(1)
        print()
        print("--------------------------------------------------")
        print("--------------   RETROPROPAGACION   --------------")
        print("--------------------------------------------------")
        print()
        m = X.shape[1]

        # Error en la capa de salida
        dS2 = self.act2 - Y  # derivada dL/dZ2
        dPesos2 = (1 / m) * np.dot(dS2, self.act1.T)
        dSesgo2 = (1 / m) * np.sum(dS2, axis=1, keepdims=True)

        # Propagar error a la capa oculta
        dAct1 = np.dot(self.pesos2.T, dS2)
        dS1 = dAct1 * derivada_sigmoide(self.act1)
        dPesos1 = (1 / m) * np.dot(dS1, X.T)
        dSesgo1 = (1 / m) * np.sum(dS1, axis=1, keepdims=True)

        # Mostrar gradientes
        print("=== Retro ===")
        print("dPesos2:\n", dPesos2)
        print("dSesgo2:\n", dSesgo2)
        print("dPesos1:\n", dPesos1)
        print("dSesgo1:\n", dSesgo1)

        # Actualizar parámetros (descenso de gradiente)
        self.pesos2 -= self.tasa * dPesos2
        self.sesgo2 -= self.tasa * dSesgo2
        self.pesos1 -= self.tasa * dPesos1
        self.sesgo1 -= self.tasa * dSesgo1

    # ======== BUCLE DE ENTRENAMIENTO ========
    def entrenar(self, X: np.ndarray, Y: np.ndarray, epocas: int = 10000, imprimir_cada: int = 100):
        """Entrena la red y muestra métricas cada `imprimir_cada` épocas."""
        # time.sleep(1)
        print()
        print("--------------------------------------------------")
        print("------------------   ENTRENAR   ------------------")
        print("--------------------------------------------------")
        print()
        loss = None
        for i in range(1, epocas + 1):
            print(f"\n--- Época {i} ---")
            Y_pred = self.adelantado(X)
            loss = self.perdida(Y_pred, Y)
            self.retropropagacion(X, Y)
            print(f"(Info) Época {i}, pérdida aprox: {loss:.6f}")
            # input("Presiona enter para siguiente")
            # time.sleep(1)
            # if i % imprimir_cada == 0:
            #     print(f"(Info) Época {i}, pérdida aprox: {loss:.6f}")


# ======== EJECUCIÓN DIRECTA ========
if __name__ == "__main__":
    # Datos XOR: entradas (2×4) y etiquetas (1×4)
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    """ 
        X       Y
    0   0       0
    0   1       1
    1   0       1
    1   1       0
    """

    red = RedNeuronal(entradas=2, ocultas=2, tasa_aprendizaje=1)
    red.entrenar(X, Y, epocas=500, imprimir_cada=1)  # Reduce epocas para ver menos salida

    print("\nPredicciones finales (redondeadas):", np.round(red.adelantado(X), 3))
    perdida_final = red.perdida(red.adelantado(X), Y)
    print("Pérdida final:", perdida_final)
    

# Temas para profundizar y referencias se mantienen al final ↓

# Temas para profundizar:
# - Álgebra lineal: multiplicación de matrices, vectores y dimensiones
# - Funciones de activación: por qué se usan (no linealidad)
# - Descenso de gradiente y tasa de aprendizaje (learning rate)
# - Backpropagation: regla de la cadena y cálculo de derivadas parciales

# Referencias:
# 1. Sigmoide y funciones de activación:
#    - Wikipedia: https://es.wikipedia.org/wiki/Funci%C3%B3n_sigmoide
#    - 3Blue1Brown (YouTube, subtítulos): https://youtu.be/xbYlEzI0SPY
# 2. Backpropagation:
#    - Rumelhart et al. (1986): https://doi.org/10.1038/323533a0
#    - Ejemplo paso a paso (Matt Mazur): https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
# 3. Entropía cruzada y optimización:
#    - Wikipedia: https://es.wikipedia.org/wiki/Entrop%C3%ADa_cruzada
# 4. Álgebra lineal para ML:
#    - Khan Academy: https://www.khanacademy.org/math/linear-algebra
#    - MIT OpenCourseWare: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/
# 5. Recursos didácticos generales:
#    - Michael Nielsen, "Neural Networks and Deep Learning": http://neuralnetworksanddeeplearning.com/
#    - Curso de Andrew Ng (Coursera): https://www.coursera.org/learn/neural-networks-deep-learning
