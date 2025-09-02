# algorithms/PCA.py
import numpy as np

class PCA:
    """
    PCA (Análisis de Componentes Principales) en NumPy puro.

    Parámetros
    ----------
    n_components : int | float | None
        - int  : nº de componentes a retener (1..d)
        - float: proporción de varianza a retener en (0, 1]; ej. 0.95 = 95%
        - None : conserva todos los componentes
    whiten : bool
        Si True, escala las proyecciones por 1/sqrt(explained_variance_) (útil para algunos modelos).
    copy : bool
        Si True, no modifica el X original en transform (buena práctica).

    Atributos tras fit()
    --------------------
    components_ : (k, d)
        Vectores de base ortonormales (cada fila es un componente).
    explained_variance_ : (k,)
        Varianza explicada por cada componente (λ_i = S_i^2 / (n-1)).
    explained_variance_ratio_ : (k,)
        Proporción de varianza explicada por cada componente.
    singular_values_ : (k,)
        Valores singulares S_i.
    mean_ : (d,)
        Media de cada feature (para centrar).
    n_components_ : int
        Nº final de componentes retenidos.
    n_features_in_ : int
        Nº de columnas de X.
    """

    def __init__(self, n_components=None, whiten=False, copy=True):
        self.n_components = n_components
        self.whiten = bool(whiten)
        self.copy = bool(copy)

        # Atributos aprendidos
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_in_ = None

    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def fit(self, X):
        X = self._check_X(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # centrar datos
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD económico: Xc = U Σ Vt  (full_matrices=False → Σ de tamaño min(n,d))
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        # varianzas de los componentes = S^2 / (n-1)
        eps = 1e-12
        denom = max(n_samples - 1, 1)
        explained_variance = (S**2) / denom
        total_var = explained_variance.sum() + eps
        explained_ratio = explained_variance / total_var

        # decidir k según n_components
        k = Vt.shape[0]
        if self.n_components is None:
            k = k
        elif isinstance(self.n_components, int):
            if not 1 <= self.n_components <= Vt.shape[0]:
                raise ValueError("n_components int fuera de rango")
            k = int(self.n_components)
        elif isinstance(self.n_components, float):
            if not (0.0 < self.n_components <= 1.0):
                raise ValueError("n_components float debe estar en (0,1]")
            # elige k mínimo tal que sum(ratio[:k]) >= target
            target = float(self.n_components)
            cumsum = np.cumsum(explained_ratio)
            k = int(np.searchsorted(cumsum, target) + 1)
        else:
            raise TypeError("n_components debe ser int, float o None")

        # recorte a k
        self.components_ = Vt[:k, :]            # (k, d)
        self.singular_values_ = S[:k]           # (k,)
        self.explained_variance_ = explained_variance[:k]
        self.explained_variance_ratio_ = explained_ratio[:k]
        self.n_components_ = k
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._check_X(X)
        Xc = X - self.mean_
        Z = Xc @ self.components_.T  # proyecciones (n, k)
        if self.whiten:
            # dividir por std de cada componente = sqrt(var)
            std = np.sqrt(self.explained_variance_)
            std[std == 0] = 1.0
            Z = Z / std
        return Z

    def inverse_transform(self, Z):
        if self.components_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z.reshape(1, -1)
        if Z.shape[1] != self.n_components_:
            raise ValueError("Z tiene nº de columnas distinto a n_components_.")
        Xc_rec = Z @ self.components_   # (n, d)
        X_rec = Xc_rec + self.mean_
        return X_rec

    def fit_transform(self, X):
        return self.fit(X).transform(X)
