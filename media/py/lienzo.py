# lienzo_mnist.py — App minimal con Tkinter para dibujar un dígito, convertirlo a 28×28
# y clasificarlo usando la red entrenada en red_mnist.py (pesos guardados en mnist_pesos.npz)
# -------------------------------------------------------------------------------------------
# Requisitos: pip install numpy pillow
# Ejecución:  python lienzo_mnist.py
# Controles:
#   - Arrastra el ratón con el botón izquierdo para dibujar.
#   - «Predecir» → procesa el dibujo y muestra el dígito estimado.
#   - «Limpiar»   → borra el canvas.
# -------------------------------------------------------------------------------------------

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# ---------------- Cargar pesos ----------------
PESOS_PATH = Path("mnist_weights.npz")
if not PESOS_PATH.exists():
    raise FileNotFoundError("No se encontró mnist_pesos.npz. Entrena primero red_mnist.py")

pesos = np.load(PESOS_PATH)
W1, b1, W2, b2 = pesos["W1"], pesos["b1"], pesos["W2"], pesos["b2"]

# ---------------- Funciones de activación ----------------

def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# ---------------- Clasificador ----------------

def predecir(im_arr: np.ndarray) -> int:
    """Recibe imagen 28×28 normalizada (784,) → retorna dígito predicho (0-9)."""
    X = im_arr.reshape(-1, 1)  # (784,1)
    # Forward manual (sin clases externas)
    A1 = sigmoide(W1 @ X + b1)
    A2 = softmax(W2 @ A1 + b2)
    return int(np.argmax(A2))

# ---------------- App Tkinter ----------------

LADO_CANVAS = 280   # 10× de 28 para mejor trazo
GROSOR_TRAZO = 20   # ancho de pincel

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Dibuja un dígito MNIST → Clasificador")

        # Canvas visible
        self.canvas = tk.Canvas(self, width=LADO_CANVAS, height=LADO_CANVAS, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=2)

        # Imagen PIL paralela donde pintamos (para obtener píxeles)
        self.im = Image.new("L", (LADO_CANVAS, LADO_CANVAS), 255)   # fondo blanco
        self.draw = ImageDraw.Draw(self.im)

        # Enlaces de evento
        self.canvas.bind("<B1-Motion>", self.pintar)

        # Botones
        tk.Button(self, text="Predecir", command=self.evaluar).grid(row=1, column=0, sticky="ew")
        tk.Button(self, text="Limpiar",  command=self.limpiar).grid(row=1, column=1, sticky="ew")

        # Etiqueta resultado
        self.resultado = tk.Label(self, text="Dibuja un dígito y presiona Predecir", font=("Arial", 14))
        self.resultado.grid(row=2, column=0, columnspan=2, pady=10)

    def pintar(self, evento):
        """Dibuja un círculo en las coordenadas del ratón (canvas y PIL)."""
        x, y = evento.x, evento.y
        r = GROSOR_TRAZO // 2
        # Canvas visible
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        # Canvas PIL para procesar
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill=0)  # 0=negro en modo 'L'

    def limpiar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, LADO_CANVAS, LADO_CANVAS], fill=255)
        self.resultado.config(text="Canvas limpiado. Dibuja otro dígito.")

    def evaluar(self):
        # 1. Escalar a 28×28 y normalizar
        im28 = self.im.resize((28, 28), Image.Resampling.LANCZOS)
        im28 = ImageOps.invert(im28)           # fondo negro, dígito blanco → invertimos
        # im28.show()
        im_arr = np.asarray(im28, dtype=np.float32).flatten() / 255.0
        # print(im_arr)

        # 2. Clasificar
        digito = predecir(im_arr)
        self.resultado.config(text=f"Predicción: {digito}")

        # 3. Debug opcional: mostrar imagen redimensionada
        # im28.resize((140,140)).show()

if __name__ == "__main__":
    App().mainloop()
