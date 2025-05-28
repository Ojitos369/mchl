import tkinter as tk
import tensorflow as tf
from rn import RedNeuronal

import numpy as np
from PIL import Image, ImageDraw, ImageOps

# ---------------- App Tkinter ----------------

LADO_CANVAS = 280   # 10× de 28 para mejor trazo
GROSOR_TRAZO = 20   # ancho de pincel

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.nn = tf.keras.models.load_model("mnist.h5")
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
        im28_inverted = ImageOps.invert(im28)           # fondo negro, dígito blanco → invertimos
        # im28.show()
        im_arr_flat = np.asarray(im28_inverted, dtype=np.float32).flatten() / 255.0
        im_arr_batch = np.expand_dims(im_arr_flat, axis=0)
        # print(im_arr)

        # 2. Clasificar
        predicciones_prob = self.nn.predict(im_arr_batch)
        print(f"Predicción: {predicciones_prob}")
        numero_predicho = np.argmax(predicciones_prob[0])
        print(f"Predicción: {numero_predicho}")
        self.resultado.config(text=f"Predicción: {numero_predicho}")

        # 3. Debug opcional: mostrar imagen redimensionada
        # im28.resize((140,140)).show()

if __name__ == "__main__":
    App().mainloop()
