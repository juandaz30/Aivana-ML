import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.ni = n_iters       #number iterations
        self.w = None           #weights
        self.b = None           #bias
        self.losses = []        #se guarda la evolución del error para mostrar cómo aprende

    def fit(self, X, y):
        """
        Entrena el modelo usando descenso de gradiente
        X: matriz de características (n_muestras x n_features)
        y: vector de etiquetas (n_muestras,)
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.ni):
            y_pred = np.dot(X, self.w) + self.b

            dw = -(2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b