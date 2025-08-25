import numpy as np

class Perceptron:
    """
    Clasificador Perceptrón (binario y multiclase por One-vs-Rest) en NumPy puro.

    API coherente con tus modelos existentes:
      - fit(X, y)
      - predict(X)
      - score(X, y)  -> accuracy

    Características:
      - Actualización online por muestras, con barajado por época (shuffle).
      - Early stopping: se detiene si no hay mejora en #errores durante 'patience' épocas,
        o si el nº de errores es 0 (datos separables).
      - Multiclase (OVR): entrena un perceptrón por clase y predice por mayor puntuación.
      - Callbacks: tras cada época, emite un dict con métricas para graficar en vivo.

    Parámetros:
      learning_rate: tamaño del paso (η). Clásico: 1.0
      n_iterations:  número de épocas (recorridos completos al dataset).
      fit_intercept: si True, usa sesgo (b).
      shuffle:       si True, baraja muestras cada época.
      random_state:  semilla para reproducibilidad (None = aleatorio).
      early_stopping:si True, aplica criterio de parada por paciencia o cero errores.
      patience:      nº de épocas sin mejora permitidas antes de detener.
      verbose:       imprime progreso cada ~10% de las épocas.
      callbacks:     lista de objetos con método .on_epoch_end(info: dict)

    Notas:
      - X: array (n_samples, n_features)
      - y: array (n_samples,) con 2 o más clases. Para binario, acepta {0,1}, {-1,1} o dos etiquetas arbitrarias.
    """

    def __init__(self,
                 learning_rate=1.0,
                 n_iterations=1000,
                 fit_intercept=True,
                 shuffle=True,
                 random_state=None,
                 early_stopping=False,
                 patience=5,
                 verbose=False,
                 callbacks=None):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.fit_intercept = bool(fit_intercept)
        self.shuffle = bool(shuffle)
        self.random_state = None if random_state is None else int(random_state)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.callbacks = [] if callbacks is None else list(callbacks)

        # Parámetros aprendidos
        self.weights = None   # binario: (n_features,); multiclase OVR: (n_classes, n_features)
        self.bias = None      # binario: escalar; multiclase OVR: (n_classes,)
        self.classes_ = None  # np.array de clases originales (ordenadas)

        # Tracking
        self.errors_history_ = []          # total por época (suma en OVR)
        self.errors_history_per_class_ = {}  # dict idx_clase -> lista de errores por época (solo multiclase)

    # ====================== helpers internos ======================
    def _maybe_reshape_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _to_pm1(self, y):
        """
        Mapea etiquetas binarias arbitrarias a {-1, +1} y devuelve:
          y_pm1, neg_label, pos_label
        """
        y = np.asarray(y)
        classes = np.unique(y)
        if classes.size != 2:
            raise ValueError("Perceptron binario requiere exactamente 2 clases.")
        neg, pos = classes.min(), classes.max()  # ordena (sirve para {0,1}, {-1,1} o arbitrarias)
        y_pm1 = np.where(y == pos, 1, -1).astype(int)
        return y_pm1, neg, pos

    def _decision_function_binary(self, X):
        return X @ self.weights + (self.bias if self.fit_intercept else 0.0)

    def decision_function(self, X):
        """
        Devuelve las puntuaciones (márgenes): 
          - binario: vector (n_samples,)
          - multiclase OVR: matriz (n_samples, n_classes), mayor = más confianza
        """
        if self.weights is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._maybe_reshape_X(X)
        if self.weights.ndim == 1:
            return self._decision_function_binary(X)
        # multiclase
        return X @ self.weights.T + (self.bias if self.fit_intercept else 0.0)

    def _epoch_indices(self, n, rng):
        return rng.permutation(n) if self.shuffle else np.arange(n)

    # ====================== entrenamiento ======================
    def fit(self, X, y):
        X = self._maybe_reshape_X(X)
        y = np.asarray(y).reshape(-1)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)

        rng = np.random.default_rng(self.random_state)
        self.errors_history_.clear()
        self.errors_history_per_class_.clear()

        if self.classes_.size == 2:
            # ---- Binario ----
            y_pm1, neg_label, pos_label = self._to_pm1(y)
            # Inicialización
            self.weights = np.zeros(n_features, dtype=float)
            self.bias = 0.0

            best_errors = np.inf
            no_improve = 0

            for epoch in range(self.n_iterations):
                idx = self._epoch_indices(n_samples, rng)
                errors = 0

                for i in idx:
                    xi = X[i]
                    yi = y_pm1[i]  # en {-1, +1}
                    a = self._decision_function_binary(xi)
                    if yi * a <= 0.0:  # clasificó mal (o en el límite)
                        self.weights += self.learning_rate * yi * xi
                        if self.fit_intercept:
                            self.bias += self.learning_rate * yi
                        errors += 1

                self.errors_history_.append(errors)

                if self.verbose and (epoch % max(1, self.n_iterations // 10) == 0):
                    print(f"[PCT] epoch={epoch:4d}  errores={errors}")

                # Callbacks (para UI)
                info = {
                    "epoch": epoch,
                    "errors": errors,
                    "weights": self.weights.copy(),
                    "bias": float(self.bias),
                    "classes": self.classes_.tolist()
                }
                for cb in self.callbacks:
                    cb.on_epoch_end(info)

                # Early stopping
                if self.early_stopping:
                    if errors == 0:
                        if self.verbose:
                            print(f"[PCT] Convergencia: 0 errores en epoch {epoch}")
                        break
                    if errors < best_errors:
                        best_errors = errors
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            if self.verbose:
                                print(f"[PCT] Parada temprana (paciencia={self.patience}) en epoch {epoch}")
                            break

        else:
            # ---- Multiclase (OVR) ----
            n_classes = self.classes_.size
            self.weights = np.zeros((n_classes, n_features), dtype=float)
            self.bias = np.zeros(n_classes, dtype=float) if self.fit_intercept else 0.0

            best_total_errors = np.inf
            no_improve = 0

            # Preconstruimos y_binarios por clase (pos = +1, resto = -1)
            Y_pm1 = []
            for k, cls in enumerate(self.classes_):
                yk = np.where(y == cls, 1, -1).astype(int)
                Y_pm1.append(yk)
                self.errors_history_per_class_[int(k)] = []

            for epoch in range(self.n_iterations):
                idx = self._epoch_indices(n_samples, rng)
                total_errors = 0

                # Entrena cada clasificador "cls vs rest" en esta época
                for k in range(n_classes):
                    errors_k = 0
                    w = self.weights[k]
                    b = self.bias[k] if self.fit_intercept else 0.0
                    yk = Y_pm1[k]

                    for i in idx:
                        xi = X[i]
                        yi = yk[i]
                        a = xi @ w + (b if self.fit_intercept else 0.0)
                        if yi * a <= 0.0:
                            w += self.learning_rate * yi * xi
                            if self.fit_intercept:
                                b += self.learning_rate * yi
                            errors_k += 1

                    # guardamos de vuelta
                    self.weights[k] = w
                    if self.fit_intercept:
                        self.bias[k] = b
                    self.errors_history_per_class_[k].append(errors_k)
                    total_errors += errors_k

                self.errors_history_.append(int(total_errors))

                if self.verbose and (epoch % max(1, self.n_iterations // 10) == 0):
                    print(f"[PCT-OVR] epoch={epoch:4d}  errores_totales={total_errors}")

                # Callbacks
                info = {
                    "epoch": epoch,
                    "errors_total": int(total_errors),
                    "errors_per_class": [self.errors_history_per_class_[k][-1] for k in range(n_classes)],
                    "weights": self.weights.copy(),
                    "bias": (self.bias.copy() if self.fit_intercept else 0.0),
                    "classes": self.classes_.tolist()
                }
                for cb in self.callbacks:
                    cb.on_epoch_end(info)

                # Early stopping
                if self.early_stopping:
                    if total_errors == 0:
                        if self.verbose:
                            print(f"[PCT-OVR] Convergencia: 0 errores en epoch {epoch}")
                        break
                    if total_errors < best_total_errors:
                        best_total_errors = total_errors
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= self.patience:
                            if self.verbose:
                                print(f"[PCT-OVR] Parada temprana (paciencia={self.patience}) en epoch {epoch}")
                            break

        return self

    # ====================== predicción y score ======================
    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._maybe_reshape_X(X)
        scores = self.decision_function(X)

        if self.classes_.size == 2:
            # binario: >= 0 => clase positiva (classes_[1]), <0 => clase negativa (classes_[0])
            y_hat = np.where(scores >= 0.0, self.classes_[1], self.classes_[0])
            return y_hat
        else:
            # multiclase: argmax por columnas
            idx = np.argmax(scores, axis=1)
            return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return float((y_pred == y).mean())
