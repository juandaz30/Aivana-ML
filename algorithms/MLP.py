import numpy as np

class MLPClassifier:
    """
    Multilayer Perceptron (clasificador) en NumPy puro.

    - Soporta binario y multiclase (softmax).
    - Descenso por lotes (batch) o mini-lotes (mini-batch SGD).
    - Regularización L2, early stopping con paciencia y validación opcional.
    - API coherente con tus otros módulos:
        fit(X, y) -> self
        predict_proba(X) -> probabilidades
        predict(X) -> etiquetas
        score(X, y) -> accuracy

    Parámetros
    ----------
    hidden_layers : tuple[int]
        Tamaño de cada capa oculta, ej. (32,), (64,32), etc.
    activation : str
        'relu' | 'tanh' | 'sigmoid' para las capas ocultas.
    learning_rate : float
        Paso del gradiente.
    n_iterations : int
        Épocas completas (si batch) o pasadas por todos los mini-lotes.
    batch_size : int | None
        Si None, usa batch completo. Si int, usa mini-lotes aleatorios de ese tamaño.
    l2 : float
        Fuerza de regularización L2 (solo en pesos, no en sesgos).
    early_stopping : bool
        Activa parada temprana en base a pérdida de validación.
    patience : int
        Épocas sin mejora antes de detener (si early_stopping=True).
    validation_split : float
        Fracción de datos a usar como validación (0..0.5 aprox.). Si 0, no separa.
    tolerance : float
        Mejora mínima en pérdida para considerar "mejora".
    decision_threshold : float
        Umbral para binario (por defecto 0.5).
    random_state : int | None
        Semilla para reproducibilidad.
    verbose : bool
        Muestra progreso cada ~10% de las iteraciones.
    callbacks : list
        Objetos con .on_epoch_end(info: dict). Útil para UI.

    Atributos tras fit()
    --------------------
    classes_ : np.ndarray
        Clases originales ordenadas.
    weights_ : list[np.ndarray]
        Pesos por capa: W_l de forma (in_l, out_l).
    biases_ : list[np.ndarray]
        Sesgos por capa: b_l de forma (out_l,).
    loss_history_ : list[float]
        Pérdida de entrenamiento por época.
    val_loss_history_ : list[float] | None
        Pérdida de validación por época (si hay split).
    n_features_in_ : int
    """

    def __init__(self,
                 hidden_layers=(32,),
                 activation='relu',
                 learning_rate=0.01,
                 n_iterations=200,
                 batch_size=None,
                 l2=0.0,
                 early_stopping=False,
                 patience=10,
                 validation_split=0.0,
                 tolerance=1e-6,
                 decision_threshold=0.5,
                 random_state=None,
                 verbose=False,
                 callbacks=None):
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.batch_size = None if batch_size is None else int(batch_size)
        self.l2 = float(l2)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.validation_split = float(validation_split)
        self.tolerance = float(tolerance)
        self.decision_threshold = float(decision_threshold)
        self.random_state = None if random_state is None else int(random_state)
        self.verbose = bool(verbose)
        self.callbacks = [] if callbacks is None else list(callbacks)

        # aprendidos
        self.classes_ = None
        self.weights_ = None
        self.biases_ = None
        self.n_features_in_ = None

        # tracking
        self.loss_history_ = []
        self.val_loss_history_ = None

    # utils
    def _ensure_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _map_y(self, y):
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("Se requieren al menos 2 clases para clasificación.")
        self.classes_ = classes
        K = classes.size
        if K == 2:
            # mapear a {0,1}: min->0, max->1
            y01 = (y == classes.max()).astype(int)
            return y01, K
        else:
            # multiclase: one-hot
            idx = np.searchsorted(classes, y)
            Y = np.eye(K)[idx]
            return Y, K

    # activaciones y derivadas
    @staticmethod
    def _relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _tanh(z):
        return np.tanh(z)

    @staticmethod
    def _tanh_deriv(z):
        t = np.tanh(z)
        return 1.0 - t * t

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    @staticmethod
    def _sigmoid_deriv(a):
        # si ya tenemos a = sigmoid(z), d/dz = a*(1-a)
        return a * (1.0 - a)

    @staticmethod
    def _softmax(z):
        # estable: z - max
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=1, keepdims=True)

    # pérdidas
    @staticmethod
    def _bce(y_true, y_prob, clip=1e-15):
        y_prob = np.clip(y_prob, clip, 1.0 - clip)
        return -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))

    @staticmethod
    def _cross_entropy(Y_true, P_pred, clip=1e-15):
        P_pred = np.clip(P_pred, clip, 1.0 - clip)
        return -np.mean(np.sum(Y_true * np.log(P_pred), axis=1))

    def _activation_pair(self):
        act = self.activation.lower()
        if act == 'relu':
            return self._relu, self._relu_deriv
        if act == 'tanh':
            return self._tanh, self._tanh_deriv
        if act == 'sigmoid':
            return self._sigmoid, self._sigmoid_deriv
        raise ValueError("activation debe ser 'relu' | 'tanh' | 'sigmoid'")

    # inicialización de pesos
    def _init_params(self, n_features, K):
        layer_sizes = [n_features] + list(self.hidden_layers) + [1 if K == 2 else K]
        self.weights_ = []
        self.biases_ = []
        rng = np.random.default_rng(self.random_state)

        hidden_act, _ = self._activation_pair()
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i < len(layer_sizes) - 2:  # capa oculta
                if hidden_act is self._relu:
                    # He init
                    W = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))
                else:
                    # Xavier
                    W = rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
            else:
                # capa de salida: Xavier suele ir bien
                W = rng.normal(0.0, np.sqrt(1.0 / fan_in), size=(fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights_.append(W)
            self.biases_.append(b)

    # forward pass: devuelve listas Z, A por capa
    def _forward(self, X, K):
        Zs = []
        As = [X]
        hidden_f, hidden_df = self._activation_pair()
        for i in range(len(self.weights_) - 1):
            Z = As[-1] @ self.weights_[i] + self.biases_[i]
            A = hidden_f(Z)
            Zs.append(Z)
            As.append(A)
        # salida
        ZL = As[-1] @ self.weights_[-1] + self.biases_[-1]
        if K == 2:
            AL = self._sigmoid(ZL)
        else:
            AL = self._softmax(ZL)
        Zs.append(ZL)
        As.append(AL)
        return Zs, As

    def _compute_loss(self, Y_true, AL, K):
        if K == 2:
            loss = self._bce(Y_true.reshape(-1, 1), AL)
        else:
            loss = self._cross_entropy(Y_true, AL)
        # L2 (solo pesos)
        if self.l2 > 0.0:
            s = sum(np.sum(W * W) for W in self.weights_)
            loss += (self.l2 / (2.0 * Y_true.shape[0])) * s
        return float(loss)

    # backward pass: devuelve gradientes dW, db
    def _backward(self, Zs, As, Y_true, K):
        n = Y_true.shape[0]
        dWs = [None] * len(self.weights_)
        dbs = [None] * len(self.biases_)

        # salida
        if K == 2:
            # As[-1] = sigmoide, Y_true en {0,1}
            dA = As[-1] - Y_true.reshape(-1, 1)  # (n,1)
        else:
            # softmax + cross-entropy: grad = P - Y
            dA = As[-1] - Y_true  # (n,K)

        # última capa
        dW = As[-2].T @ dA / n
        db = dA.mean(axis=0)
        if self.l2 > 0.0:
            dW += (self.l2 / n) * self.weights_[-1]
        dWs[-1] = dW
        dbs[-1] = db

        # capas ocultas (hacia atrás)
        hidden_f, hidden_df = self._activation_pair()
        dZ_next = dA
        for i in range(len(self.weights_) - 2, -1, -1):
            # derivar a través de capa i
            # dA_i = dZ_{i+1} @ W_{i+1}^T
            dA_i = dZ_next @ self.weights_[i + 1].T
            # dZ_i = dA_i * f'(Z_i)
            dZ_i = dA_i * hidden_df(Zs[i])
            dW_i = As[i].T @ dZ_i / n
            db_i = dZ_i.mean(axis=0)
            if self.l2 > 0.0:
                dW_i += (self.l2 / n) * self.weights_[i]
            dWs[i] = dW_i
            dbs[i] = db_i
            dZ_next = dZ_i

        return dWs, dbs

    def _iterate_minibatches(self, X, Y, batch_size, rng):
        n = X.shape[0]
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sel = idx[start:end]
            yield X[sel], (Y[sel] if Y is not None else None)

    # ============================ API ============================
    def fit(self, X, y):
        X = self._ensure_2d(X)
        Y, K = self._map_y(y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self._init_params(n_features, K)

        # split validación
        use_val = self.validation_split > 0.0
        if use_val:
            rng = np.random.default_rng(self.random_state)
            n_val = max(1, int(n_samples * self.validation_split))
            idx = rng.permutation(n_samples)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, Y_tr = X[tr_idx], (Y[tr_idx])
            X_val, Y_val = X[val_idx], (Y[val_idx])
            self.val_loss_history_ = []
        else:
            X_tr, Y_tr = X, Y
            X_val = Y_val = None

        self.loss_history_.clear()
        best_val = np.inf
        no_improve = 0
        rng = np.random.default_rng(self.random_state)

        for epoch in range(self.n_iterations):
            if self.batch_size is None:
                # batch completo
                Zs, As = self._forward(X_tr, K)
                loss = self._compute_loss(Y_tr, As[-1], K)
                dWs, dbs = self._backward(Zs, As, Y_tr, K)

                # actualización
                for i in range(len(self.weights_)):
                    self.weights_[i] -= self.learning_rate * dWs[i]
                    self.biases_[i]  -= self.learning_rate * dbs[i]
            else:
                # mini-batch SGD
                total_loss = 0.0
                total_batches = 0
                for Xb, Yb in self._iterate_minibatches(X_tr, Y_tr, self.batch_size, rng):
                    Zs, As = self._forward(Xb, K)
                    total_loss += self._compute_loss(Yb, As[-1], K)
                    total_batches += 1
                    dWs, dbs = self._backward(Zs, As, Yb, K)
                    for i in range(len(self.weights_)):
                        self.weights_[i] -= self.learning_rate * dWs[i]
                        self.biases_[i]  -= self.learning_rate * dbs[i]
                loss = total_loss / max(1, total_batches)

            self.loss_history_.append(float(loss))

            # validación
            if use_val:
                Zs_val, As_val = self._forward(X_val, K)
                vloss = self._compute_loss(Y_val, As_val[-1], K)
                self.val_loss_history_.append(float(vloss))

            # verbose
            if self.verbose and (epoch % max(1, self.n_iterations // 10) == 0):
                if use_val:
                    print(f"[MLP] epoch={epoch:4d}  loss={loss:.6f}  val_loss={vloss:.6f}")
                else:
                    print(f"[MLP] epoch={epoch:4d}  loss={loss:.6f}")

            # callbacks
            info = {
                "epoch": epoch,
                "loss": float(loss),
                "val_loss": (float(vloss) if use_val else None),
                "weights_norm": float(sum(np.linalg.norm(W) for W in self.weights_)),
            }
            for cb in self.callbacks:
                cb.on_epoch_end(info)

            # early stopping
            if self.early_stopping and use_val:
                if epoch > 0:
                    improved = (self.val_loss_history_[-2] - self.val_loss_history_[-1]) > self.tolerance
                else:
                    improved = True
                if improved:
                    best_val = self.val_loss_history_[-1]
                    no_improve = 0
                    # (opcional) podrías guardar copia de los mejores pesos
                    best_weights = [W.copy() for W in self.weights_]
                    best_biases = [b.copy() for b in self.biases_]
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        # restaurar mejores pesos
                        self.weights_ = best_weights
                        self.biases_ = best_biases
                        if self.verbose:
                            print(f"[MLP] Early stopping en epoch {epoch} (paciencia={self.patience})")
                        break

        return self

    def predict_proba(self, X):
        if self.weights_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._ensure_2d(X)
        K = 2 if self.classes_.size == 2 else self.classes_.size
        _, As = self._forward(X, K)
        AL = As[-1]
        if K == 2:
            # devolver prob de la clase positiva (classes_[-1])
            return AL.reshape(-1)
        return AL  # (n,K)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.classes_.size == 2:
            yhat01 = (proba >= self.decision_threshold).astype(int)
            return np.where(yhat01 == 1, self.classes_[-1], self.classes_[0])
        else:
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return float((y_pred == y).mean())
