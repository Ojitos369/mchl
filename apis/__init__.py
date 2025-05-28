import json
import math
import os
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

# ... (tus funciones _tanh, _sigmoid, _relu, _softmax y el diccionario _ACTIVATIONS se mantienen igual) ...
# ... (la clase _DenseLayer se mantiene igual) ...

# --------------------------------------------------
# Funciones de activación y derivadas
# --------------------------------------------------

def _tanh(z):
    return np.tanh(z)

def _tanh_d(a):
    return 1.0 - np.square(a)

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def _sigmoid_d(a):
    return a * (1.0 - a)

def _relu(z):
    return np.maximum(0.0, z)

def _relu_d(a):
    return (a > 0.0).astype(a.dtype)

def _softmax(z):
    z_shift = z - np.max(z, axis=0, keepdims=True)  # estabilidad numérica
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def _softmax_d(a):
    raise NotImplementedError("La derivada exacta de softmax no es necesaria; se usa simplificación con cross‑entropy.")

_ACTIVATIONS: dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]] = {
    "tanh": {"f": _tanh, "fp": _tanh_d},
    "sigmoid": {"f": _sigmoid, "fp": _sigmoid_d},
    "relu": {"f": _relu, "fp": _relu_d},
    "softmax": {"f": _softmax, "fp": _softmax_d},
}

# --------------------------------------------------
# Capa densamente conectada
# --------------------------------------------------
class _DenseLayer:
    def __init__(self, n_in: int, n_out: int, activation: str = "relu", seed: int | None = None):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Función de activación desconocida: {activation}")
        rng = np.random.default_rng(seed)
        if activation == "relu":
            std = math.sqrt(2 / n_in)
            self.b = np.full((n_out, 1), 0.01)
        elif activation == "softmax":
            std = math.sqrt(1 / n_in)
            self.b = np.zeros((n_out, 1))
        else:
            std = math.sqrt(1 / n_in)
            self.b = np.zeros((n_out, 1))
        self.W = rng.normal(0.0, std, (n_out, n_in))
        self.f = _ACTIVATIONS[activation]["f"]
        self.fp = _ACTIVATIONS[activation]["fp"]
        self.activation_name = activation
        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        self.Z = self.W @ A_prev + self.b
        self.A = self.f(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, A_prev: np.ndarray, loss: str) -> np.ndarray:
        m = A_prev.shape[1]
        if self.activation_name == "softmax" and loss == "cross_entropy":
            dZ = dA
        else:
            dZ = dA * self.fp(self.A)
        self.dW = (dZ @ A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        return self.W.T @ dZ

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def to_dict(self):
        return {
            "W": self.W.tolist(),
            "b": self.b.squeeze().tolist(),
            "activation": self.activation_name,
        }

    @classmethod
    def from_dict(cls, data: dict):
        layer = cls(1, 1)
        layer.W = np.array(data["W"], dtype=float)
        layer.b = np.array(data["b"], dtype=float).reshape(-1, 1)
        layer.activation_name = data["activation"]
        layer.f = _ACTIVATIONS[layer.activation_name]["f"]
        layer.fp = _ACTIVATIONS[layer.activation_name]["fp"]
        return layer

# --------------------------------------------------
# Red neuronal completa
# --------------------------------------------------
class RedNeuronal:
    def __init__(self, layer_sizes: Sequence[int], *, activations: Sequence[str] | None = None,
                 lr: float = 0.01, seed: int | None = None):
        if len(layer_sizes) < 2:
            raise ValueError("Debe haber al menos una capa de entrada y una de salida.")
        if activations is None:
            activations = ["relu"] * (len(layer_sizes) - 2) + ["sigmoid"]
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("activations debe tener len(layer_sizes)-1 elementos.")
        self.layers: List[_DenseLayer] = []
        # `layer_sizes[0]` es el número de características de entrada
        # `layer_sizes[i]` es el número de salidas de la capa anterior (o entradas de la actual)
        # `layer_sizes[i+1]` es el número de neuronas (y salidas) de la capa actual
        for i in range(len(layer_sizes) - 1):
            self.layers.append(_DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i], seed))
        self.lr = lr
        self.history: list[float] = []

    @staticmethod
    def _mse(A: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        m = Y.shape[1] # Y tiene forma (n_outputs, m)
        loss = np.mean(np.square(A - Y)) # A tiene forma (n_outputs, m)
        dA = 2 * (A - Y) / m
        return loss, dA

    @staticmethod
    def _cross_entropy(A: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        m = Y.shape[1] # Y tiene forma (n_outputs, m)
        eps = 1e-12
        loss = -np.sum(Y * np.log(A + eps)) / m # A tiene forma (n_outputs, m)
        dA = (A - Y) / m
        return loss, dA

    def _forward(self, X: np.ndarray) -> np.ndarray:
        # X ya está en formato (n_features, m)
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A # Retorna A en formato (n_outputs_final_layer, m)

    def fit(self, X: np.ndarray, Y: np.ndarray, *, epochs: int = 1000, batch_size: int = 32,
            loss: str = "mse", verbose: int = 100, patience: int | None = None,
            threshold: float | None = None, shuffle: bool = True):
        if loss not in {"mse", "cross_entropy"}:
            raise ValueError("Pérdida soportada: 'mse' o 'cross_entropy'")
        
        # X se espera como (n_features_input, m_samples)
        # Y se espera como (n_features_output, m_samples)
        # No se necesita transponer X ni Y si ya vienen en este formato.
        X_fit = X.astype(float)
        Y_fit = Y.astype(float)

        # Validar dimensiones con la arquitectura de la red
        # layer_sizes[0] es el número de características de entrada
        if X_fit.shape[0] != self.layers[0].W.shape[1]:
            raise ValueError(f"El número de características de entrada X ({X_fit.shape[0]}) "
                             f"no coincide con el tamaño de entrada de la primera capa ({self.layers[0].W.shape[1]}).")
        # layer_sizes[-1] es el número de salidas de la red
        if Y_fit.shape[0] != self.layers[-1].W.shape[0]:
            raise ValueError(f"El número de características de salida Y ({Y_fit.shape[0]}) "
                             f"no coincide con el tamaño de salida de la última capa ({self.layers[-1].W.shape[0]}).")

        m = X_fit.shape[1] # Número de muestras
        best_loss: float | None = None
        epochs_without_improve = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                perm = np.random.permutation(m)
                X_perm = X_fit[:, perm]
                Y_perm = Y_fit[:, perm]
            else:
                X_perm, Y_perm = X_fit, Y_fit

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_perm[:, start:end] # (n_features_input, batch_m)
                Y_batch = Y_perm[:, start:end] # (n_features_output, batch_m)

                A = X_batch
                # Cache de A_prev para backprop necesita el formato (features, m_batch)
                # El A de la capa anterior (o X_batch para la primera capa)
                # se almacena en las propias capas durante el forward pass.
                # O mejor, pasamos A_prev explícitamente.
                # El forward pass interno ya guarda A en cada capa.
                
                # Guardar las entradas de cada capa para el backward pass.
                # self.layers[0].A_input = X_batch # A_input no existe en _DenseLayer
                # La entrada a la primera capa es X_batch
                # La entrada a la capa i+1 es self.layers[i].A

                A_final = self._forward(X_batch) # A_final es (n_outputs, batch_m)

                if loss == "mse":
                    l, dA = self._mse(A_final, Y_batch)
                else: # cross_entropy
                    l, dA = self._cross_entropy(A_final, Y_batch)
                # dA tiene forma (n_outputs, batch_m)

                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    # A_prev es la activación de la capa anterior,
                    # o X_batch si es la primera capa oculta.
                    A_prev_layer = X_batch if i == 0 else self.layers[i-1].A
                    dA = layer.backward(dA, A_prev_layer, loss) # dA se actualiza para la capa anterior

                for layer in self.layers:
                    layer.update(self.lr)

            A_epoch = self._forward(X_fit) # Evaluar con todas las muestras (sin permutar)
            if loss == "mse":
                epoch_loss, _ = self._mse(A_epoch, Y_fit)
            else:
                epoch_loss, _ = self._cross_entropy(A_epoch, Y_fit)
            self.history.append(epoch_loss)
            
            self.last_loss_str = f"{epoch_loss:.6f}"

            if verbose and (epoch % verbose == 0 or epoch == 1 or epoch == epochs):
                print(f"Época {epoch:>6}/{epochs} | pérdida={epoch_loss:.6f}")

            if threshold is not None and epoch_loss <= threshold:
                print(f"🎯 Umbral {threshold} alcanzado en época {epoch}")
                break
            if patience is not None:
                if best_loss is None or epoch_loss < best_loss - 1e-8: # Añadir pequeña tolerancia para mejora
                    best_loss = epoch_loss
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        print(f"⏹️  Sin mejora en {patience} épocas. Deteniendo en época {epoch}.")
                        break

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X se espera en formato (n_features_input, m_samples)
        # Validar dimensiones
        if X.shape[0] != self.layers[0].W.shape[1]:
             raise ValueError(f"El número de características de entrada X ({X.shape[0]}) "
                             f"no coincide con el tamaño de entrada de la primera capa ({self.layers[0].W.shape[1]}).")
        X_pred = X.astype(float)
        # _forward espera (n_features, m) y devuelve (n_outputs_final_layer, m)
        A_final = self._forward(X_pred)
        return A_final # Devuelve en formato (n_outputs, m_samples)

    def save(self, file_path: str | os.PathLike):
        data = {
            "layers": [layer.to_dict() for layer in self.layers],
            "lr": self.lr,
            # Guardar layer_sizes podría ser útil para reconstruir con from_dict
            "layer_sizes": [self.layers[0].W.shape[1]] + [layer.W.shape[0] for layer in self.layers]
        }
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Pesos guardados en '{file_path}'.")

    @classmethod
    def load(cls, file_path: str | os.PathLike):
        with open(file_path) as f:
            data = json.load(f)
        
        layers_data = data["layers"]
        
        # Reconstruir layer_sizes y activations a partir de los datos guardados
        # Es más robusto si se guardan explícitamente
        if "layer_sizes" in data:
            layer_sizes = data["layer_sizes"]
        else: # Inferir si no está guardado (menos robusto si la estructura cambia)
            n_input = np.array(layers_data[0]["W"]).shape[1]
            layer_sizes = [n_input] + [np.array(ld["W"]).shape[0] for ld in layers_data]

        activations = [ld["activation"] for ld in layers_data]
        
        # Crear instancia temporal y luego reemplazar sus capas.
        # Usar layer_sizes y activations para el constructor.
        # El primer tamaño en layer_sizes es el de entrada, luego las salidas de cada capa.
        rn = cls(layer_sizes, activations=activations, lr=data.get("lr", 0.01))
        
        # Sobrescribir las capas con las cargadas (ya que el constructor las inicializó)
        rn.layers = []
        for ld in layers_data:
            rn.layers.append(_DenseLayer.from_dict(ld))
        
        # Asegurar que lr también se cargue correctamente incluso si se inicializó
        rn.lr = data.get("lr", 0.01)
        print(f"🧠 Pesos cargados desde '{file_path}'.")
        return rn

