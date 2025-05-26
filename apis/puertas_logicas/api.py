# Python
import os
import numpy as np
import json

# Ojitos369
from ojitos369.utils import get_d

# User
from app.core.bases.apis import PostApi, GetApi
from app.settings import MEDIA_DIR
from .neural_network import NeuralNetwork

class HelloWorld(GetApi):
    def main(self):
        self.response = {
            'message': 'Hello World'
        }

class Train(PostApi):
    def main(self):
        trainded_weights = MEDIA_DIR + "/npz/pl.npz"
        if os.path.exists(trainded_weights):
            os.remove(trainded_weights)

        x = self.data["x"]
        y = self.data["y"]
        lr = float(self.data.get("lr", 0.5))
        af = self.data.get("af", "sig")
        pasos = int(self.data.get("pasos", 500000))
        print(f"x: {x}")
        print(f"y: {y}")
        print(f"lr: {lr}")
        print(f"af: {af}")
        print(f"pasos: {pasos}")

        X = np.array(x)
        Y = np.array(y)
        print(f"X: {X}")
        print(f"Y: {Y}")
        nn = NeuralNetwork(input_size=2, hidden_size=2, learning_rate=lr, save_weights=trainded_weights, activation_function=af)
        nn.train(X, Y, epochs=pasos, print_every=10000)
        ultima_perdida = nn.last_loss_str
        print(f"ultima_perdida: {ultima_perdida}")
        
        self.response = {
            "message": f"Network Trained. {ultima_perdida}"
        }


class Calculate(GetApi):
    def main(self):
        trainded_weights = MEDIA_DIR + "/npz/pl.npz"
        if not os.path.exists(trainded_weights):
            raise self.MYE("Network not trained")
        
        x1 = int(self.data["x1"])
        x2 = int(self.data["x2"])
        af = self.data.get("af", "sig")

        X = np.array([[x1], [x2]])
        
        nn = NeuralNetwork(input_size=2, hidden_size=2, learning_rate=0.5, activation_function=af)
        nn.load_weights(trainded_weights)
        y = nn.predict(X)
        self.response = {
            "respuesta": round(y[0][0], 3)
        }


