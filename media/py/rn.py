import json
import math
import os
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from numba import jit # ¡Importamos a nuestro amigo Numba!

# --------------------------------------------------
# Tipo de dato para cálculos numéricos
# --------------------------------------------------
DTYPE = np.float32 # Usaremos float32 para optimizar memoria y velocidad

# --------------------------------------------------
# Funciones de activación y derivadas (Optimizadas con Numba)
# --------------------------------------------------

@jit(nopython=True, cache=True) # nopython=True para máxima velocidad, cache=True para recompilar menos
def _tanh_nb(z: np.ndarray) -> np.ndarray:
    return np.tanh(z.astype(DTYPE)) # Asegurar DTYPE interno en Numba

@jit(nopython=True, cache=True)
def _tanh_d_nb(a: np.ndarray) -> np.ndarray:
    return (1.0 - np.square(a)).astype(DTYPE)

@jit(nopython=True, cache=True)
def _sigmoid_nb(z: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-z.astype(DTYPE)))).astype(DTYPE)

@jit(nopython=True, cache=True)
def _sigmoid_d_nb(a: np.ndarray) -> np.ndarray:
    return (a * (1.0 - a)).astype(DTYPE)

@jit(nopython=True, cache=True)
def _relu_nb(z: np.ndarray) -> np.ndarray:
    return np.maximum(DTYPE(0.0), z.astype(DTYPE))

@jit(nopython=True, cache=True)
def _relu_d_nb(a: np.ndarray) -> np.ndarray:
    return (a > DTYPE(0.0)).astype(DTYPE)


# Softmax es un poco más compleja para Numba nopython=True debido a axis y keepdims
# en np.max y np.sum. NumPy puro es eficiente.
def _softmax(z: np.ndarray) -> np.ndarray:
    z_typed = z.astype(DTYPE)
    z_shift = z_typed - np.max(z_typed, axis=0, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def _softmax_d(a: np.ndarray) -> np.ndarray:
    raise NotImplementedError("La derivada exacta de softmax no es necesaria; se usa simplificación con cross‑entropy.")

_ACTIVATIONS: dict[str, dict[str, Callable[[np.ndarray], np.ndarray]]] = {
    "tanh": {"f": _tanh_nb, "fp": _tanh_d_nb},
    "sigmoid": {"f": _sigmoid_nb, "fp": _sigmoid_d_nb},
    "relu": {"f": _relu_nb, "fp": _relu_d_nb},
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
            self.b = np.full((n_out, 1), 0.01, dtype=DTYPE)
        elif activation == "softmax":
            std = math.sqrt(1 / n_in)
            self.b = np.zeros((n_out, 1), dtype=DTYPE)
        else:
            std = math.sqrt(1 / n_in)
            self.b = np.zeros((n_out, 1), dtype=DTYPE)

        self.W = rng.normal(0.0, std, (n_out, n_in)).astype(DTYPE)
        self.f = _ACTIVATIONS[activation]["f"]
        self.fp = _ACTIVATIONS[activation]["fp"]
        self.activation_name = activation

        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        self.Z = (self.W @ A_prev) + self.b
        self.A = self.f(self.Z)
        return self.A

    def backward(self, dA: np.ndarray, A_prev: np.ndarray, loss: str) -> np.ndarray:
        m = A_prev.shape[1]
        dZ: np.ndarray
        if self.activation_name == "softmax" and loss == "cross_entropy":
            dZ = dA.astype(DTYPE)
        else:
            dZ = dA * self.fp(self.A)
        
        self.dW = (dZ @ A_prev.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        return (self.W.T @ dZ).astype(DTYPE)

    def update(self, lr: float):
        # lr es float, se promocionará a DTYPE si es necesario durante la operación
        self.W -= DTYPE(lr) * self.dW
        self.b -= DTYPE(lr) * self.db


    def to_dict(self):
        return {
            "W": self.W.astype(float).tolist(),
            "b": self.b.astype(float).squeeze().tolist(), # squeeze() puede devolver un escalar si b es (1,1)
            "activation": self.activation_name,
        }

    @classmethod
    def from_dict(cls, data: dict):
        temp_W = np.array(data["W"])
        n_out, n_in = temp_W.shape
        
        layer = cls(n_in, n_out, activation=data["activation"]) 
        
        layer.W = np.array(data["W"], dtype=DTYPE)
        # Asegurar que b sea siempre 2D (N,1)
        b_array = np.array(data["b"], dtype=DTYPE)
        if b_array.ndim == 0: # Si era un escalar por squeeze() de un (1,1)
            layer.b = b_array.reshape(1,1)
        elif b_array.ndim == 1: # Si era un vector 1D
            layer.b = b_array.reshape(-1, 1)
        else: # Ya era 2D
            layer.b = b_array
            
        return layer

# --------------------------------------------------
# Red neuronal completa
# --------------------------------------------------
class RedNeuronal:
    def __init__(self, layer_sizes: Sequence[int], *, activations: Sequence[str] | None = None,
                 lr: float = 0.01, seed: int | None = None):
        if len(layer_sizes) < 2:
            raise ValueError("Debe haber al menos una capa de entrada y una de salida.")
        
        default_activations = ["relu"] * (len(layer_sizes) - 2) + ["sigmoid"]
        if activations is None:
            activations = default_activations
        elif len(activations) == 1 and len(layer_sizes) -1 > 1 : # Si se provee una sola para múltiples capas
             # Si la única activación provista es para la capa de salida (caso de red de 1 capa)
             if len(layer_sizes) -1 == 1:
                 pass # Se usa la activación provista
             else: # Se replica para las ocultas y se usa la default para la salida, o la misma si solo hay ocultas
                num_hidden_layers = len(layer_sizes) - 2
                if num_hidden_layers > 0:
                    final_activation = activations[0] if len(default_activations) == num_hidden_layers else default_activations[-1]
                    activations = [activations[0]] * num_hidden_layers + [final_activation]
                # Si no hay capas ocultas, activations ya tiene 1 elemento para la capa de salida
        
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"Activations debe tener {len(layer_sizes)-1} elementos "
                f"(se proveyeron {len(activations)}, se esperaban para la estructura {layer_sizes}). "
                f"Activaciones actuales: {activations}"
            )
            
        self.layers: List[_DenseLayer] = []
        current_seed = seed
        for i in range(len(layer_sizes) - 1):
            layer_seed = current_seed + i if current_seed is not None else None
            self.layers.append(
                _DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i], layer_seed)
            )
        self.lr = DTYPE(lr)
        self.history: list[float] = []
        self.last_loss_str: str = "N/A"

    @staticmethod
    def _mse(A: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        m = Y.shape[1]
        diff = A - Y
        loss = np.mean(np.square(diff))
        dA = (DTYPE(2.0) * diff) / m
        return float(loss), dA.astype(DTYPE)

    @staticmethod
    def _cross_entropy(A: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
        m = Y.shape[1]
        eps = DTYPE(1e-7) 
        # Clip A para evitar log(0) o log(>1) si Y no es one-hot y A puede ser >1
        A_clipped = np.clip(A, eps, 1.0 - eps if np.any(A > 1.0 - eps) else np.inf)

        log_A = np.log(A_clipped) # A_clipped ya es DTYPE por A y eps
        loss = -np.sum(Y * log_A) / m
        dA = (A - Y) / m # Correcto cuando se combina con la capa softmax que espera dZ = A - Y
        return float(loss), dA.astype(DTYPE)


    def _forward(self, X: np.ndarray) -> np.ndarray:
        A = X 
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def fit(self, X: np.ndarray, Y: np.ndarray, *, epochs: int = 1000, batch_size: int = 32,
            loss: str = "mse", verbose: int = 100, patience: int | None = None,
            threshold: float | None = None, shuffle: bool = True):

        if loss not in {"mse", "cross_entropy"}:
            raise ValueError("Pérdida soportada: 'mse' o 'cross_entropy'")
        
        X_fit = X.astype(DTYPE)
        Y_fit = Y.astype(DTYPE) # Y también debe ser DTYPE

        if X_fit.shape[0] != self.layers[0].W.shape[1]:
            raise ValueError(
                f"El número de características de entrada X ({X_fit.shape[0]}) "
                f"no coincide con el tamaño de entrada de la primera capa ({self.layers[0].W.shape[1]})."
            )
        if Y_fit.shape[0] != self.layers[-1].W.shape[0]:
             raise ValueError(
                f"El número de características de salida Y ({Y_fit.shape[0]}) "
                f"no coincide con el tamaño de salida de la última capa ({self.layers[-1].W.shape[0]})."
            )

        m = X_fit.shape[1]
        best_loss: float | None = None
        epochs_without_improve = 0
        
        loss_func = self._mse if loss == "mse" else self._cross_entropy

        for epoch in range(1, epochs + 1):
            if shuffle:
                perm = np.random.permutation(m)
                X_perm = X_fit[:, perm]
                Y_perm = Y_fit[:, perm]
            else:
                X_perm, Y_perm = X_fit, Y_fit

            for start in range(0, m, batch_size):
                end = min(start + batch_size, m) # Asegurar que end no exceda m
                X_batch = X_perm[:, start:end]
                Y_batch = Y_perm[:, start:end]

                A_final = self._forward(X_batch)
                
                _, dA = loss_func(A_final, Y_batch)

                for i in reversed(range(len(self.layers))):
                    layer = self.layers[i]
                    A_prev_layer = X_batch if i == 0 else self.layers[i-1].A
                    dA = layer.backward(dA, A_prev_layer, loss)

                for layer in self.layers:
                    layer.update(self.lr)
            
            A_epoch_full = self._forward(X_fit)
            epoch_loss_val, _ = loss_func(A_epoch_full, Y_fit)
            self.history.append(epoch_loss_val)
            self.last_loss_str = f"{epoch_loss_val:.6f}"

            if verbose and (epoch % verbose == 0 or epoch == 1 or epoch == epochs):
                print(f"Época {epoch:>6}/{epochs} | pérdida={epoch_loss_val:.6f}")

            if threshold is not None and epoch_loss_val <= threshold:
                print(f"🎯 Umbral {threshold:.6f} alcanzado en época {epoch} con pérdida {epoch_loss_val:.6f}")
                break
            if patience is not None:
                if best_loss is None or epoch_loss_val < best_loss - DTYPE(1e-7):
                    best_loss = epoch_loss_val
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        print(f"⏹️  Sin mejora en {patience} épocas. Deteniendo en época {epoch}. Mejor pérdida: {best_loss:.6f}")
                        break
        # Opcional: Limpiar cachés después del entrenamiento
        # for layer in self.layers:
        #     layer.Z, layer.A, layer.dW, layer.db = None, None, None, None


    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1: # Si es un solo ejemplo (vector 1D)
            X = X.reshape(-1, 1) # Convertir a (n_features, 1)
        if X.shape[0] != self.layers[0].W.shape[1]:
             raise ValueError(
                f"El número de características de entrada X ({X.shape[0]}) "
                f"no coincide con el tamaño de entrada de la primera capa ({self.layers[0].W.shape[1]})."
            )
        X_pred = X.astype(DTYPE)
        A_final = self._forward(X_pred)
        return A_final

    def save(self, file_path: str | os.PathLike):
        file_p = Path(file_path)
        file_p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "layer_sizes": [self.layers[0].W.shape[1]] + [layer.W.shape[0] for layer in self.layers],
            "activations": [layer.activation_name for layer in self.layers],
            "lr": float(self.lr),
            "layers_params": [layer.to_dict() for layer in self.layers]
        }
        with open(file_p, "w") as f:
            json.dump(data, f, indent=4)
        print(f"💾 Pesos guardados en '{file_p}'.")

    @classmethod
    def load(cls, file_path: str | os.PathLike):
        file_p = Path(file_path)
        if not file_p.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {file_p}")
        with open(file_p, "r") as f:
            data = json.load(f)
        
        layer_sizes = data["layer_sizes"]
        activations = data["activations"] 
        lr = data.get("lr", 0.01) 

        rn = cls(layer_sizes, activations=activations, lr=lr) # seed no se guarda
        
        if len(rn.layers) != len(data["layers_params"]):
            raise ValueError(
                f"Inconsistencia en el número de capas. "
                f"Esperadas por arquitectura: {len(rn.layers)}, "
                f"Encontradas en archivo: {len(data['layers_params'])}"
            )

        for i, layer_params_dict in enumerate(data["layers_params"]):
            # La capa rn.layers[i] ya existe, solo actualizamos sus pesos W y b
            # y verificamos la activación.
            expected_activation = rn.layers[i].activation_name
            loaded_activation = layer_params_dict["activation"]
            if expected_activation != loaded_activation:
                 print(
                    f"Advertencia: Discrepancia de activación para capa {i}. "
                    f"Esperada por arquitectura: '{expected_activation}', Cargada del archivo: '{loaded_activation}'. "
                    "Se utilizará la activación definida por la arquitectura de la red al cargar ('{expected_activation}')."
                )
            
            # Cargamos W y b usando el método from_dict de _DenseLayer para asegurar el DTYPE
            # y el formato correcto, pero solo tomamos W y b del objeto temporal.
            temp_layer_from_dict = _DenseLayer.from_dict(layer_params_dict)
            rn.layers[i].W = temp_layer_from_dict.W
            rn.layers[i].b = temp_layer_from_dict.b
            
        print(f"🧠 Pesos cargados desde '{file_p}'.")
        return rn
