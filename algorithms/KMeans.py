import numpy as np

class KMeans:
    """
    K-means desde cero (NumPy puro).
    - init: 'k-means++' (recomendado) o 'random'
    - n_init: reinicios con distintas semillas; me quedo con la mejor inercia
    - score(X): devuelve -inertia (más alto es mejor), coherente con sklearn

    Atributos tras fit():
      cluster_centers_: (k, d)
      labels_:          (n,)
      inertia_:         float (SSE final)
      n_iter_:          int   (iteraciones de la mejor corrida)
      inertia_history_: lista de inercias por época (mejor corrida)
      centers_history_: lista de centros por época (mejor corrida)

    Callbacks: en cada época se emite {"run", "epoch", "inertia", "centers"}.
    """
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, random_state=None,
                 verbose=False, callbacks=None):
        self.n_clusters = int(n_clusters)
        self.init = init
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = None if random_state is None else int(random_state)
        self.verbose = bool(verbose)
        self.callbacks = [] if callbacks is None else list(callbacks)

        # parámetros aprendidos
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

        # tracking para UI
        self.inertia_history_ = None
        self.centers_history_ = None

    # ---------- helpers ----------
    def _check_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _squared_euclidean(self, X, centers):
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c   (vectorizado y estable)
        Xn = np.sum(X**2, axis=1, keepdims=True)           # (n,1)
        Cn = np.sum(centers**2, axis=1, keepdims=True).T   # (1,k)
        d2 = Xn + Cn - 2.0 * (X @ centers.T)               # (n,k)
        d2[d2 < 0] = 0.0  # por redondeo numérico
        return d2

    def _init_random(self, X, rng):
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError("n_clusters no puede ser mayor que n_samples.")
        idx = rng.choice(n, size=self.n_clusters, replace=False)
        return X[idx].copy()

    def _init_kmeanspp(self, X, rng):
        n, d = X.shape
        if self.n_clusters > n:
            raise ValueError("n_clusters no puede ser mayor que n_samples.")
        centers = np.empty((self.n_clusters, d), dtype=float)
        # primer centro aleatorio
        idx0 = rng.integers(n)
        centers[0] = X[idx0]
        # distancias^2 mínimas a centros ya elegidos
        d2 = np.sum((X - centers[0])**2, axis=1)
        for c in range(1, self.n_clusters):
            total = d2.sum()
            if not np.isfinite(total) or total <= 0:
                # degenerado: todos iguales; completa aleatorio
                rest = rng.choice(n, size=self.n_clusters - c, replace=False)
                centers[c:] = X[rest]
                break
            probs = d2 / total
            idx = rng.choice(n, p=probs)
            centers[c] = X[idx]
            new_d2 = np.sum((X - centers[c])**2, axis=1)
            d2 = np.minimum(d2, new_d2)
        return centers

    def _assign_labels_inertia(self, X, centers):
        d2 = self._squared_euclidean(X, centers)  # (n,k)
        labels = np.argmin(d2, axis=1)
        inertia = float(d2[np.arange(X.shape[0]), labels].sum())
        return labels, inertia, d2

    def _compute_centers(self, X, labels):
        k, d = self.n_clusters, X.shape[1]
        centers = np.zeros((k, d), dtype=float)
        counts = np.bincount(labels, minlength=k).astype(int)
        for j in range(k):
            if counts[j] > 0:
                centers[j] = X[labels == j].mean(axis=0)
            else:
                centers[j] = np.nan  # marcar vacío
        return centers, counts

    # ---------- entrenamiento ----------
    def fit(self, X):
        X = self._check_X(X)
        n, d = X.shape
        rng_global = np.random.default_rng(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_hist_inertia = None
        best_hist_centers = None
        best_n_iter = None

        for run in range(self.n_init):
            rng = np.random.default_rng(rng_global.integers(1 << 32))
            if self.init == 'k-means++':
                centers = self._init_kmeanspp(X, rng)
            elif self.init == 'random':
                centers = self._init_random(X, rng)
            else:
                raise ValueError("init debe ser 'k-means++' o 'random'.")

            hist_inertia, hist_centers = [], []
            prev = centers

            for it in range(self.max_iter):
                labels, inertia, d2 = self._assign_labels_inertia(X, prev)
                centers_new, counts = self._compute_centers(X, labels)

                # clusters vacíos: reubicar en puntos más alejados
                if np.isnan(centers_new).any():
                    dmin = d2[np.arange(n), labels]
                    empties = np.where(np.isnan(centers_new).any(axis=1))[0]
                    m = len(empties)
                    far_idx = np.argpartition(dmin, -m)[-m:]
                    centers_new[empties] = X[far_idx]

                shift = np.linalg.norm(centers_new - prev, axis=1)
                max_shift = float(np.max(shift))
                hist_inertia.append(inertia)
                hist_centers.append(centers_new.copy())

                if self.verbose and (it % max(1, self.max_iter // 10) == 0):
                    print(f"[KMEANS] run={run:02d} iter={it:03d} inertia={inertia:.6f} max_shift={max_shift:.6f}")

                info = {"run": run, "epoch": it, "inertia": inertia, "centers": centers_new.copy()}
                for cb in self.callbacks:
                    cb.on_epoch_end(info)

                prev = centers_new
                if max_shift <= self.tol:
                    break

            labels, inertia, _ = self._assign_labels_inertia(X, prev)
            n_iter = it + 1

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = prev.copy()
                best_labels = labels.copy()
                best_hist_inertia = hist_inertia
                best_hist_centers = hist_centers
                best_n_iter = n_iter

        # guardar mejor solución
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = float(best_inertia)
        self.n_iter_ = int(best_n_iter)
        self.inertia_history_ = list(best_hist_inertia)
        self.centers_history_ = [c.copy() for c in best_hist_centers]
        return self

    # ---------- uso ----------
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._check_X(X)
        labels, _, _ = self._assign_labels_inertia(X, self.cluster_centers_)
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_

    def transform(self, X, squared=False):
        """Distancias de cada punto a cada centro (n,k). Si squared=True, devuelve distancias^2."""
        if self.cluster_centers_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._check_X(X)
        d2 = self._squared_euclidean(X, self.cluster_centers_)
        return d2 if squared else np.sqrt(d2)

    def score(self, X):
        """Devuelve -inertia (coherente con sklearn): más alto es mejor."""
        X = self._check_X(X)
        _, inertia, _ = self._assign_labels_inertia(X, self.cluster_centers_)
        return -float(inertia)
