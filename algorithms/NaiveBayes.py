import numpy as np

class NaiveBayes:
    """
    Naive Bayes desde cero (NumPy), con tres variantes:
      - 'gaussian'    : para features continuas (modelo Gaussiano por clase/feature)
      - 'multinomial' : para conteos/no-negativos (texto, bolsas de palabras)
      - 'bernoulli'   : para binario {0,1} (o se binariza con un umbral)

    API coherente con tus modelos:
      fit(X, y) -> self
      predict_proba(X) -> probabilidades
      predict(X) -> etiquetas
      score(X, y) -> accuracy

    Parámetros
    ----------
    nb_type : str
        'gaussian' | 'multinomial' | 'bernoulli'
    var_smoothing : float
        (solo gaussian) suaviza varianzas: var += var_smoothing * var_total
    alpha : float
        (multinomial/bernoulli) suavizado de Laplace/additivo >= 0
    class_priors : array-like | None
        Priors P(y=c). Si None, usa frecuencia empírica.
    binarize : float | None
        (bernoulli) si no es None, convierte X a 0/1 con X > binarize
    decision_threshold : float
        Umbral para binario en predict() (por defecto 0.5)
    """

    def __init__(self,
                 nb_type='gaussian',
                 var_smoothing=1e-9,
                 alpha=1.0,
                 class_priors=None,
                 binarize=None,
                 decision_threshold=0.5):
        self.nb_type = str(nb_type).lower()
        if self.nb_type not in ('gaussian', 'multinomial', 'bernoulli'):
            raise ValueError("nb_type debe ser 'gaussian' | 'multinomial' | 'bernoulli'")
        self.var_smoothing = float(var_smoothing)
        self.alpha = float(alpha)
        if self.alpha < 0:
            raise ValueError("alpha debe ser >= 0")
        self.class_priors = None if class_priors is None else np.asarray(class_priors, dtype=float)
        self.binarize = binarize if binarize is None else float(binarize)
        self.decision_threshold = float(decision_threshold)

        # aprendidos
        self.classes_ = None            # (K,)
        self.class_count_ = None        # (K,)
        self.class_prior_ = None        # (K,)
        self.n_features_in_ = None

        # gaussian
        self.theta_ = None              # medias por clase/feature (K, d)
        self.var_ = None                # varianzas por clase/feature (K, d)

        # multinomial
        self.feature_count_ = None      # (K, d)
        self.feature_log_prob_ = None   # (K, d)

        # bernoulli
        self.feature_prob_ = None       # (K, d)
        self.neg_feature_prob_ = None   # (K, d) para log(1-p)

    # -------------------- helpers --------------------
    def _ensure_2d(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _map_y(self, y):
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        idx = np.searchsorted(self.classes_, y)
        return idx, K

    def _check_multinomial_input(self, X):
        if np.any(X < 0):
            raise ValueError("MultinomialNB requiere X >= 0 (conteos o no-negativos).")

    def _maybe_binarize(self, X):
        if self.binarize is not None:
            return (X > self.binarize).astype(float)
        # Si ya es {0,1} no toca; si no, asumimos que ya viene binario
        return X

    # -------------------- entrenamiento --------------------
    def fit(self, X, y):
        X = self._ensure_2d(X)
        y_idx, K = self._map_y(y)
        n, d = X.shape
        self.n_features_in_ = d

        # priors
        self.class_count_ = np.bincount(y_idx, minlength=K).astype(float)
        if self.class_priors is None:
            self.class_prior_ = self.class_count_ / max(1.0, self.class_count_.sum())
        else:
            pri = np.asarray(self.class_priors, dtype=float)
            if pri.shape[0] != K:
                raise ValueError("class_priors debe tener tamaño K (nº de clases)")
            if (pri < 0).any() or not np.isclose(pri.sum(), 1.0):
                raise ValueError("class_priors deben ser >=0 y sumar 1")
            self.class_prior_ = pri

        if self.nb_type == 'gaussian':
            # medias y varianzas por clase/feature
            self.theta_ = np.zeros((K, d), dtype=float)
            self.var_ = np.zeros((K, d), dtype=float)
            for k in range(K):
                Xk = X[y_idx == k]
                if Xk.shape[0] == 0:
                    # clase sin muestras (raro): dejar medias/var por defecto
                    continue
                mu = Xk.mean(axis=0)
                var = Xk.var(axis=0)  # ddof=0 (MLE)
                # suavizado de varianza
                total_var = X.var(axis=0).mean() if np.isfinite(X.var()).any() else 0.0
                var += self.var_smoothing * total_var
                # evitar var=0 exacta
                var[var == 0] = self.var_smoothing
                self.theta_[k] = mu
                self.var_[k] = var

        elif self.nb_type == 'multinomial':
            self._check_multinomial_input(X)
            # conteos por clase/feature
            self.feature_count_ = np.zeros((K, d), dtype=float)
            for k in range(K):
                Xk = X[y_idx == k]
                self.feature_count_[k] = Xk.sum(axis=0)
            # prob(feature|clase) con suavizado alpha
            smoothed_fc = self.feature_count_ + self.alpha
            smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
            self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

        else:  # bernoulli
            Xb = self._maybe_binarize(X)
            # p(x_j=1 | y=c)
            self.feature_prob_ = np.zeros((K, d), dtype=float)
            for k in range(K):
                Xk = Xb[y_idx == k]
                # suavizado alfa: (sum + alpha) / (n_k + 2*alpha)
                nk = Xk.shape[0]
                pk = (Xk.sum(axis=0) + self.alpha) / max(1.0, (nk + 2.0 * self.alpha))
                # clamp leve para estabilidad
                pk = np.clip(pk, 1e-12, 1 - 1e-12)
                self.feature_prob_[k] = pk
            self.neg_feature_prob_ = 1.0 - self.feature_prob_

        return self

    # -------------------- predicción --------------------
    def _joint_log_likelihood_gaussian(self, X):
        # log P(y=c) + sum_j log N(x_j | mu_{c,j}, var_{c,j})
        K, d = self.theta_.shape
        # términos gaussianos en log: -0.5*(log(2pi*var) + (x-mu)^2/var)
        log_prior = np.log(self.class_prior_ + 1e-15)
        jll = np.zeros((X.shape[0], K), dtype=float)
        for k in range(K):
            mu = self.theta_[k]
            var = self.var_[k]
            # log coeficiente
            log_coef = -0.5 * (np.log(2.0 * np.pi * var)).sum()
            # distancia cuadrática escalada
            diff = X - mu
            quad = -0.5 * ((diff * diff) / var).sum(axis=1)
            jll[:, k] = log_prior[k] + log_coef + quad
        return jll

    def _joint_log_likelihood_multinomial(self, X):
        # X debe ser no-negativa; usar log P(x|y) = sum_j x_j * log p_{j|y}
        self._check_multinomial_input(X)
        log_prior = np.log(self.class_prior_ + 1e-15)
        return X @ self.feature_log_prob_.T + log_prior

    def _joint_log_likelihood_bernoulli(self, X):
        Xb = self._maybe_binarize(X)
        log_prior = np.log(self.class_prior_ + 1e-15)
        log_p = np.log(self.feature_prob_)
        log_q = np.log(self.neg_feature_prob_)
        # log P(x|y) = sum_j [ x_j*log p + (1-x_j)*log (1-p) ]
        ll_pos = Xb @ log_p.T
        ll_neg = (1.0 - Xb) @ log_q.T
        return ll_pos + ll_neg + log_prior

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._ensure_2d(X)
        if self.nb_type == 'gaussian':
            jll = self._joint_log_likelihood_gaussian(X)
        elif self.nb_type == 'multinomial':
            jll = self._joint_log_likelihood_multinomial(X)
        else:
            jll = self._joint_log_likelihood_bernoulli(X)
        # softmax en log-space para probabilidades
        jll = jll - jll.max(axis=1, keepdims=True)
        np.exp(jll, out=jll)
        probs = jll / jll.sum(axis=1, keepdims=True)
        if self.classes_.size == 2:
            # devolver prob de la clase positiva (classes_[-1]) como vector (n,)
            pos_idx = 1  # porque classes_ está ordenada
            return probs[:, pos_idx]
        return probs

    def predict(self, X):
        if self.classes_ is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit primero.")
        X = self._ensure_2d(X)
        if self.nb_type == 'gaussian':
            jll = self._joint_log_likelihood_gaussian(X)
        elif self.nb_type == 'multinomial':
            jll = self._joint_log_likelihood_multinomial(X)
        else:
            jll = self._joint_log_likelihood_bernoulli(X)
        # argmax por clase
        idx = np.argmax(jll, axis=1)
        if self.classes_.size == 2 and self.decision_threshold != 0.5:
            # si el usuario fijó un umbral distinto, usar predict_proba
            proba = self.predict_proba(X)
            yhat01 = (proba >= self.decision_threshold).astype(int)
            return np.where(yhat01 == 1, self.classes_[1], self.classes_[0])
        return self.classes_[idx]

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        return float((y_pred == y).mean( ))
