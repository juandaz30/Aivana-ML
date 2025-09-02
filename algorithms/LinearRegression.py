import numpy as np

class LinearRegression:

    # Regresión Lineal desde cero

    def __init__(self,
                 learning_rate=0.01,    # qué tan fuerte ajusta los pesos en cada paso (si es muy grande, salta; si es muy pequeño, avanza lento)
                 n_iterations=1000,     # cuántos pasos máximos va a dar el entrenamiento.
                 tolerance=1e-8,        # si la diferencia entre una iteración y la siguiente es menor a este valor, ya convergió, entre más pequeño más preciso
                 early_stopping=False,  # True lo detiene cuando la mejora es menor a tolerance, False hace todo el número de iteraciones 
                 verbose=False,         # True imprime progreso del entrenamiento (para debug)
                 fit_intercept=True):   # True indica que el modelo incluye un bias(b), False indica un paso por 0,0 de la recta
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.tolerance = float(tolerance)
        self.early_stopping = bool(early_stopping)
        self.verbose = bool(verbose)
        self.fit_intercept = bool(fit_intercept)

        # parámetros aprendidos
        self.weights = None
        self.bias = None

        # tracking
        self.loss_history = []

    # ---------- helpers comunes ----------
    def _initialize_params(self, n_features):
        # inicializa pesos y sesgo (pequeño aleatorio en w, b=0)
        # randn() genera una matriz del tamaño de n_features con números aleatorios cercanos a 0
        self.weights = np.random.randn(n_features) * 0.01 
        self.bias = 0.0

    def _predict_raw(self, X):
        #predicción lineal Xw + b de cada iteración. fit_intercept controla el uso de bias
        #multiplicación matricial entre X y los pesos y luego se le suma la bias
        return X @ self.weights + (self.bias if self.fit_intercept else 0.0)

    @staticmethod
    # mean squared error
    def _mse(y, y_pred):
        #calcula la diferencia (o error) entre cada valor predicho y su valor real correspondiente
        #se eleva al cuadrado para asegurar + y para penalizar errores grandes. la ecuación mean() calcula el promedio de todos los errores
        return np.mean((y - y_pred) ** 2)

    # entrenación del modelo con gradiente descendente
    def fit(self, X, y):

        # se asegura que los datos estén en el formato correcto
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        #variables con la plantilla de X. n_samples: número de muestras, n_features: número de variables ind
        n_samples, n_features = X.shape
        #se generan valores (arreglos del tamaño de X) para peso y bias
        self._initialize_params(n_features)
        self.loss_history.clear()

        #itera
        for i in range(self.n_iterations):
            # forward
            # PRIMERA PREDICCIÓN: sobre los pesos y bias (0 por ahora) generados
            y_pred = self._predict_raw(X)

            # CORE del algoritmo
            # +(le falta a la predicción) -(se pasó la predicción), db y dw tendrá el signo opuesto
            residual = (y - y_pred) 
            # promedio del error, indica cuánto y en qué dirección se debe ajustar cada peso para reducir el error
            dw = -(2.0 / n_samples) * (X.T @ residual)
            # indica el ajuste de cesgos (bias)
            db = -(2.0 / n_samples) * residual.sum() if self.fit_intercept else 0.0

            # update
            # ajusta el o los pesos según la taza de aprendizaje y dw (corrección), se resta si dw es positivo (se pasó), se suma si dw es negativo (le falta)
            self.weights -= self.learning_rate * dw
            # ajusta el cesgo según la taza de aprendizaje y db (corrección)
            if self.fit_intercept:
                self.bias -= self.learning_rate * db

            # loss e historial: guarda el error total por cada iteración y lo agrega al historial
            loss = self._mse(y, y_pred)
            self.loss_history.append(loss)

            # muestra información cada iteración si verbose es True, cada self.n_iterations // 10 iteraciones
            if self.verbose and (i % max(1, self.n_iterations // 10) == 0):
                print(f"[GD] iter={i:4d}  MSE={loss:.6f}")

            # early stopping (por mejora mínima en pérdida)
            if self.early_stopping and i > 0:
                # si ya se alcanzó el umbral de tolerancia se sale
                if abs(self.loss_history[-2] - self.loss_history[-1]) < self.tolerance:
                    if self.verbose:
                        print(f"[GD] Convergencia en la iteración {i}  (ΔMSE < {self.tolerance})")
                    return self
        return self

    # predicciones con modelo ya entenado
    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        # se asegura que los datos estén en el formato correcto
        X = np.asarray(X, dtype=float)
        return self._predict_raw(X)

    # calcula el coeficiente de determinación R^2
    def score(self, X, y):
        """R² = 1 - SS_res/SS_tot"""
        # se asegura que los datos estén en el formato correcto
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X)
        # suma de los errores al cuadrado entre los valores reales y los predichos.
        ss_res = np.sum((y - y_pred) ** 2)
        # suma de los errores al cuadrado entre los valores reales y su media.
        ss_tot = np.sum((y - y.mean()) ** 2)
        # si devuelve cercano a 1, el modelo es buenísimo, si devuelve cercano a 0, es terrible
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0