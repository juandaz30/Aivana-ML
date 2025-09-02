# algorithms/DecisionTreeClassifier.py
import numpy as np

class DecisionTreeClassifier:
    """
    Árbol de Decisión (CART) para clasificación en NumPy puro.

    - Soporta multiclase.
    - Funciones de impureza: 'gini' (por defecto) o 'entropy'.
    - Búsqueda eficiente de umbrales: por cada feature, ordena X[:, j] una vez
      y evalúa todos los cortes posibles donde cambia el valor.
    - Criterios de parada: max_depth, min_samples_split, min_samples_leaf,
      pureza (unanimidad) o ganancia no positiva.
    - max_features: None (todas), 'sqrt', 'log2', int o float en (0,1].

    API:
      fit(X, y) -> self
      predict(X) -> (n,)
      predict_proba(X) -> (n, K)
      score(X, y) -> accuracy

    Atributos tras fit:
      n_classes_ : int
      classes_ : np.ndarray (etiquetas originales ordenadas)
      n_features_in_ : int
      feature_importances_ : (d,) suma normalizada de reducciones de impureza por feature
      tree_ : estructura interna de nodos (clase _Node)

    Nota: entrada numérica. Para categóricas, usar OneHotEncoder externo.
    """

    class _Node:
        __slots__ = ("is_leaf", "pred_class", "proba", "feature", "threshold",
                     "left", "right", "n_samples", "impurity", "impurity_decrease")
        def __init__(self):
            self.is_leaf = True
            self.pred_class = None      # int (índice en [0..K-1])
            self.proba = None           # (K,)
            self.feature = None         # int
            self.threshold = None       # float
            self.left = None            # _Node
            self.right = None           # _Node
            self.n_samples = 0
            self.impurity = 0.0
            self.impurity_decrease = 0.0

    def __init__(self,
                 criterion="gini",            # 'gini' | 'entropy'
                 max_depth=None,              # int | None
                 min_samples_split=2,         # int
                 min_samples_leaf=1,          # int
                 max_features=None,           # None|'sqrt'|'log2'|int|float
                 random_state=None,
                 verbose=False):
        self.criterion = criterion
        self.max_depth = None if max_depth is None else int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = None if random_state is None else int(random_state)
        self.verbose = bool(verbose)

        # rellenados en fit
        self.n_classes_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.tree_ = None

        # acumulador de importancias (suma de reducciones de impureza por feature)
        self._impurity_importance_sum = None

    # -------------------- utils internos --------------------
    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _map_y(self, y):
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.size
        y_idx = np.searchsorted(self.classes_, y)  # mapea a {0..K-1}
        return y_idx

    def _max_features_to_use(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(d)))
            if mf == "log2":
                return max(1, int(np.log2(d)))
            raise ValueError("max_features string debe ser 'sqrt' o 'log2'")
        if isinstance(mf, int):
            return max(1, min(d, mf))
        if isinstance(mf, float):
            if not (0 < mf <= 1.0):
                raise ValueError("max_features float en (0,1]")
            return max(1, int(np.ceil(mf * d)))
        raise TypeError("max_features debe ser None|'sqrt'|'log2'|int|float")

    # Impurezas
    @staticmethod
    def _gini(counts):
        # counts: (K,)
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        return 1.0 - np.sum(p * p)

    @staticmethod
    def _entropy(counts):
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    def _impurity(self, counts):
        return self._gini(counts) if self.criterion == "gini" else self._entropy(counts)

    # -------------------- búsqueda de mejor split --------------------
    def _best_split_for_feature(self, x_col, y_idx, total_counts):
        """
        x_col: (n,) valores de una feature
        y_idx: (n,) clases en {0..K-1}
        total_counts: (K,) conteo total del nodo actual
        Devuelve: best_gain, best_threshold (float), left_counts (K,), right_counts (K,)
        Si no hay mejora, best_gain = 0.0 y threshold = None
        """
        n = x_col.shape[0]
        # ordenar por feature
        order = np.argsort(x_col, kind="mergesort")  # estable
        x_sorted = x_col[order]
        y_sorted = y_idx[order]

        # puntos candidatos: cortes entre valores distintos
        # usamos cumsum de indicadores de clase para tener conteos a izquierda
        K = self.n_classes_
        # matriz de indicadores acumulados: para eficiencia, hacemos una sola pasada
        # one-hot implícito vía indexing
        left_counts = np.zeros((n, K), dtype=np.int64)
        for i in range(n):
            c = y_sorted[i]
            left_counts[i] = left_counts[i-1] if i > 0 else 0
            left_counts[i, c] += 1

        # evaluar en posiciones donde x_sorted[i] != x_sorted[i+1]
        parent_imp = self._impurity(total_counts)
        best_gain = 0.0
        best_thr = None
        best_left = None
        best_right = None

        # para respetar min_samples_leaf
        min_leaf = self.min_samples_leaf
        for i in range(n - 1):
            if x_sorted[i] == x_sorted[i+1]:
                continue
            n_left = i + 1
            n_right = n - n_left
            if n_left < min_leaf or n_right < min_leaf:
                continue

            lc = left_counts[i]                            # (K,)
            rc = total_counts - lc                         # (K,)
            imp_left = self._impurity(lc)
            imp_right = self._impurity(rc)
            # reducción de impureza (CART):
            # gain = parent_imp - (nL/n)*impL - (nR/n)*impR
            gain = parent_imp - (n_left/n)*imp_left - (n_right/n)*imp_right

            if gain > best_gain:
                best_gain = float(gain)
                # umbral = punto medio entre valores consecutivos
                best_thr = 0.5 * (x_sorted[i] + x_sorted[i+1])
                best_left = lc.copy()
                best_right = rc.copy()

        if best_thr is None:
            return 0.0, None, None, None
        return best_gain, float(best_thr), best_left, best_right

    def _best_split(self, X, y_idx):
        """
        Devuelve: feature_idx, threshold, left_mask, right_mask, gain
        o (None, None, None, None, 0.0) si no hay mejora.
        """
        n, d = X.shape
        total_counts = np.bincount(y_idx, minlength=self.n_classes_).astype(np.int64)
        if total_counts.max() == n:
            return None, None, None, None, 0.0  # nodo puro

        # seleccionar subconjunto de features si aplica
        rng = np.random.default_rng(self.random_state)
        n_feat = self._max_features_to_use(d)
        feats = np.arange(d)
        if n_feat < d:
            feats = rng.choice(d, size=n_feat, replace=False)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_counts = None
        best_right_counts = None

        for j in feats:
            gain, thr, lc, rc = self._best_split_for_feature(X[:, j], y_idx, total_counts)
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, j, thr
                best_left_counts, best_right_counts = lc, rc

        if best_feature is None:
            return None, None, None, None, 0.0

        # construir máscaras para partir datos según el umbral hallado
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        return best_feature, best_threshold, left_mask, right_mask, best_gain

    # -------------------- construcción recursiva --------------------
    def _build(self, X, y_idx, depth):
        node = self._Node()
        node.n_samples = X.shape[0]
        counts = np.bincount(y_idx, minlength=self.n_classes_).astype(np.int64)
        node.impurity = self._impurity(counts)
        node.pred_class = int(np.argmax(counts))
        proba = counts / counts.sum() if counts.sum() > 0 else np.ones(self.n_classes_) / self.n_classes_
        node.proba = proba

        # criterios de parada
        if (self.max_depth is not None and depth >= self.max_depth) \
           or node.n_samples < self.min_samples_split \
           or counts.max() == node.n_samples:
            node.is_leaf = True
            return node

        # buscar mejor split
        feat, thr, Lmask, Rmask, gain = self._best_split(X, y_idx)
        if feat is None or gain <= 0.0:
            node.is_leaf = True
            return node

        # comprobar min_samples_leaf
        if Lmask.sum() < self.min_samples_leaf or Rmask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            return node

        # asignar split
        node.is_leaf = False
        node.feature = int(feat)
        node.threshold = float(thr)
        node.impurity_decrease = float(gain)

        if self.verbose:
            print(f"[TREE] depth={depth} feat={node.feature} thr={node.threshold:.6f} "
                  f"gain={node.impurity_decrease:.6f} nL={Lmask.sum()} nR={Rmask.sum()}")

        # actualizar importancia por feature (acumulada, ponderada por n_samples)
        self._impurity_importance_sum[node.feature] += gain * node.n_samples

        # recursión
        node.left = self._build(X[Lmask], y_idx[Lmask], depth + 1)
        node.right = self._build(X[Rmask], y_idx[Rmask], depth + 1)
        return node

    # -------------------- API pública --------------------
    def fit(self, X, y):
        X = self._check_X(X)
        y_idx = self._map_y(y)
        self.n_features_in_ = X.shape[1]
        self._impurity_importance_sum = np.zeros(self.n_features_in_, dtype=float)

        self.tree = None  # retrocompat
        self.tree_ = self._build(X, y_idx, depth=0)

        # normalizar importancias
        s = self._impurity_importance_sum.sum()
        if s > 0:
            self.feature_importances_ = self._impurity_importance_sum / s
        else:
            self.feature_importances_ = np.zeros_like(self._impurity_importance_sum)
        return self

    def _predict_row(self, x):
        node = self.tree_
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.pred_class, node.proba

    def predict_proba(self, X):
        if self.tree_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._check_X(X)
        n = X.shape[0]
        P = np.zeros((n, self.n_classes_), dtype=float)
        for i in range(n):
            _, proba = self._predict_row(X[i])
            P[i] = proba
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        y_hat = self.predict(X)
        return float((y_hat == y).mean())
