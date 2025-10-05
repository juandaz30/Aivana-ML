import numpy as np

class LogisticRegression:
    """
    Regresión Logística binaria desde cero (batch Gradient Descent).
    - Optimiza la pérdida logarítmica (binary cross-entropy).
    - Vectorizada, con early stopping y regularización L2 opcional.

    Uso típico:
    -----------
    model = LogisticRegression(learning_rate=0.1, n_iterations=2000, early_stopping=True)
    model.fit(X_train, y_train)              # y ∈ {0,1} (si viene en {-1,1} lo convierte)
    y_prob = model.predict_proba(X_test)     # probabilidades P(y=1|x)
    y_pred = model.predict(X_test)           # etiquetas 0/1 por umbral
    acc    = model.score(X_test, y_test)     # accuracy
    """

    def __init__(self,
                 learning_rate=0.1,        # tamaño del paso de gradiente
                 n_iterations=2000,        # iteraciones máximas
                 tolerance=1e-6,           # mejora mínima de pérdida para considerar convergencia
                 early_stopping=False,     # si la diferencia entre una iteración y la siguiente es menor a este valor, ya convergió, entre más pequeño más preciso
                 verbose=False,            # True: imprime progreso
                 fit_intercept=True,       # True: usa término de sesgo (bias)
                 l2=0.0,                   # fuerza de regularización L2 (0.0 = sin regularización), evita overfitting haciendo que los coeficientes o pesos sean mas pequeños
                 decision_threshold=0.5,   # umbral o criterio para convertir probas a clases. clase 1 o clase 0 (si el sigmoide >= decision_threshold (0.5) es clase 1 si es < es clase 0)
                 clip=1e-15):              # para estabilidad numérica. evita que log() copalse por valores = 0 o = 1
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tolerance = float(tolerance)
        self.early_stopping = bool(early_stopping)
        self.verbose = bool(verbose)
        self.fit_intercept = bool(fit_intercept)
        self.l2 = float(l2)
        self.decision_threshold = float(decision_threshold)
        self.clip = float(clip)

        # parámetros aprendidos
        self.weights = None
        self.bias = None 

        # tracking
        self.loss_history = []
        # Etiquetas originales (neg, pos) para que predict devuelva las mismas
        self.classes_ = None 

    # ---------- helpers ----------

    def _ensure_2d(self, X):
        """
        Asegura que X sea 2D (n_samples, n_features).
        Si viene 1D (n,), lo convertimos a (n,1) para evitar errores tontos.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
    
    # toma lo calculado por el modelo y lo combierte a una probabilidad entre 0 y 1
    # si z es muy negativo se acerca a 0, si es muy positivo se acerca a 1
    @staticmethod
    def _sigmoid(z):
        # evita overflow para |z| grande
        # Sigmoid numéricamente estable
        """ Fórmula:
          si z >= 0:  σ(z) = 1 / (1 + exp(-z))
          si z <  0:  σ(z) = exp(z) / (1 + exp(z))
        Con esto evitamos exp(500) o exp(-500) que rompe la máquina.
        """
        z = np.asarray(z, dtype=float)
        # Para estabilidad, manejar positivos/negativos por separado
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        neg = ~pos
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        expz = np.exp(z[neg])
        out[neg] = expz / (1.0 + expz)
        return out

    # calcula la perdida para la regresión logística, si la predicción es perfecta devuelve 0 y entre más se equivoque más grande es el valor
    @staticmethod
    def _bce_loss(y, p, clip=1e-15):
        # Binary cross-entropy: - mean [ y log(p) + (1-y) log(1-p) ]
        p = np.clip(p, clip, 1.0 - clip)
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    #un vector con tantos valores como features se tengan. Se inicializan muy cercanos a 0, con cada iteración estos valores se van ajustando
    def _initialize_params(self, n_features):
        # w ~ N(0, 0.01^2), b = 0
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    # producto punto entre los datos de entrada, los pesos aprendidos y se le suma la bias
    # son los datos en crudo, se puede decir que es cuanto "suma" cada dato de entrada al resultado según sus features
    def _predict_logits(self, X):
        return X @ self.weights + (self.bias if self.fit_intercept else 0.0)


    #convierte los datos crudos de _predict_logits en intervalos entre 0 y 1 con sigmoide
    def _predict_proba_internal(self, X):
        return self._sigmoid(self._predict_logits(X))

    @staticmethod
    def _ensure_binary_labels(y):
        """Devuelve y en {0,1}. Si viene en {-1,1}, lo mapea a {0,1}.
        Si hay más de dos clases, lanza error."""
        y = np.asarray(y).reshape(-1)
        classes = np.unique(y)
        if classes.size == 2:
            # Si las clases son {-1,1}, mapear a {0,1}
            if np.allclose(classes, [-1, 1]):
                y01 = (y == 1).astype(float)  # 1 -> 1, -1 -> 0
                return y01, -1, 1
            # Si ya son {0,1}, simplemente devolver
            if np.allclose(classes, [0, 1]):
                return y.astype(float), 0, 1
            # Si son dos valores arbitrarios, mapear min->0, max->1
            y01 = (y == classes.max()).astype(float)
            return y01, classes.min(), classes.max()
        elif classes.size == 1:
            # Caso degenerado: todas las etiquetas iguales
            # (el modelo igual entrena, pero score es trivial)
            if classes[0] in (0, 1):
                return y.astype(float), 0, 1
            y01 = np.zeros_like(y, dtype=float)  # todo a 0
            return y01, classes[0], None
        # Si tiene más de 2 valores distintos arroja un error
        else:
            raise ValueError("LogisticRegression binaria requiere exactamente 2 clases en y.")

    # ---------- entrenamiento ----------
    def fit(self, X, y):
        """
        Entrena el modelo con gradiente descendente (batch).
        - X: (n_samples, n_features)
        - y: (n_samples,) en {0,1} (si viene en {-1,1} se convierte internamente)
        """
        X = np.asarray(X, dtype=float)
        y, self._y_neg, self._y_pos = self._ensure_binary_labels(y)

        n_samples, n_features = X.shape
        self._initialize_params(n_features)
        self.loss_history.clear()

        for i in range(self.n_iterations):
            # forward: logits -> probabilidades
            p = self._predict_proba_internal(X)

            # pérdida (con regularización L2 solo en w, no en b)
            loss = self._bce_loss(y, p, clip=self.clip)
            if self.l2 > 0.0:
                loss += (self.l2 / (2.0 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

            # gradientes
            # residual = (p - y)  → gradiente del BCE
            residual = (p - y)  # shape (n_samples,)
            dw = (X.T @ residual) / n_samples   # shape (n_features,)
            if self.l2 > 0.0:
                dw += (self.l2 / n_samples) * self.weights  # regularización L2
            db = residual.mean() if self.fit_intercept else 0.0

            # actualización
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # verbose
            if self.verbose and (i % max(1, self.n_iterations // 10) == 0):
                print(f"[LOG] iter={i:4d}  loss={loss:.6f}")

            # early stopping: compara mejoras en loss
            if self.early_stopping and i > 0:
                if abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
                    if self.verbose:
                        print(f"[LOG] Convergencia en iter {i} (Δloss < {self.tolerance})")
                    break

        return self

    # ---------- predicción ----------
    def predict_proba(self, X):
        """Devuelve P(y=1|x) como vector de probabilidades en [0,1]."""
        if self.weights is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = np.asarray(X, dtype=float)
        return self._predict_proba_internal(X)

    def predict(self, X):
        """Devuelve etiquetas {0,1} usando self.decision_threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.decision_threshold).astype(int)

    def score(self, X, y):
        """Accuracy: proporción de aciertos."""
        y = np.asarray(y).reshape(-1)
        # Mapear y a {0,1} usando el mismo criterio que en fit
        y_mapped, _, _ = self._ensure_binary_labels(y)
        y_pred = self.predict(X).astype(float)
        return float((y_pred == y_mapped).mean())