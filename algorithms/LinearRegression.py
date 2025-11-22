# algorithms/LinearRegression.py
import numpy as np

class LinearRegression:
    def __init__(self,
                 learning_rate=0.01,
                 n_iterations=1000,
                 fit_intercept=True,
                 early_stopping=True,
                 tolerance=1e-6,
                 patience=10,
                 normalize=True,
                 max_grad_norm=1e6,
                 verbose=False):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.fit_intercept = bool(fit_intercept)
        self.early_stopping = bool(early_stopping)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.normalize = bool(normalize)
        self.max_grad_norm = float(max_grad_norm)
        self.verbose = bool(verbose)

        # Se llenan en fit()
        self.weights = None
        self.x_mean_ = None
        self.x_std_  = None
        self.y_mean_ = None
        self.loss_history_ = []

    @staticmethod
    def _safe_mse(y_true, y_pred):
        # cálculo en float64 y con clipping suave para evitar overflow numérico
        err = (y_true - y_pred).astype(np.float64)
        # si hay valores exagerados, recorta antes de cuadrar
        err = np.clip(err, -1e12, 1e12)
        return float(np.mean(err * err))

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        n = X.shape[0]
        return np.hstack([np.ones((n, 1), dtype=X.dtype), X])

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        # --- Normalización de X y centrado de y (mejora estabilidad) ---
        if self.normalize:
            self.x_mean_ = X.mean(axis=0)
            self.x_std_  = X.std(axis=0)
            # evitar división por cero
            self.x_std_[self.x_std_ == 0] = 1.0
            Xn = (X - self.x_mean_) / self.x_std_
        else:
            self.x_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.x_std_  = np.ones(X.shape[1], dtype=np.float64)
            Xn = X

        self.y_mean_ = y.mean()
        yc = y - self.y_mean_  # centrar objetivo

        Xn = self._add_intercept(Xn)

        n_samples, n_features = Xn.shape
        self.weights = np.zeros(n_features, dtype=np.float64)

        best_loss = np.inf
        wait = 0
        self.loss_history_.clear()

        for it in range(self.n_iterations):
            y_pred = Xn @ self.weights
            loss = self._safe_mse(yc, y_pred)
            self.loss_history_.append(float(loss))

            if self.verbose and (it % max(1, self.n_iterations // 10) == 0):
                print(f"[LR] iter={it} mse={loss:.6f}")

            # Early stopping por no-mejora
            if self.early_stopping:
                if loss + self.tolerance < best_loss:
                    best_loss = loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        if self.verbose:
                            print(f"[LR] early stop @ iter {it}, best_mse={best_loss:.6f}")
                        break

            # Gradiente (2/N) X^T (y_pred - y)
            residual = (y_pred - yc)
            # clipping de gradiente para evitar explosiones numéricas
            grad = (2.0 / n_samples) * (Xn.T @ residual)
            norm = np.linalg.norm(grad)
            if np.isfinite(norm) and norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / (norm + 1e-12))

            # update
            self.weights -= self.learning_rate * grad

            # corta si se volvió inestable
            if not np.all(np.isfinite(self.weights)) or not np.isfinite(loss):
                # intenta un “step back” sencillo bajando el LR
                self.learning_rate *= 0.1
                if self.verbose:
                    print("[LR] inestabilidad detectada, reduciendo learning_rate y continuando")
                # revertir el último paso aproximando (sube un poco)
                self.weights += self.learning_rate * grad
                # si sigue mal, salimos
                if not np.all(np.isfinite(self.weights)):
                    break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # aplicar la misma normalización
        Xn = (X - self.x_mean_) / self.x_std_ if self.normalize else X
        Xn = self._add_intercept(Xn)
        y_pred_centered = Xn @ self.weights
        # deshacer el centrado de y
        return y_pred_centered + self.y_mean_
