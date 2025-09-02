import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.linear_model import LogisticRegression as SklearnLogR
from sklearn.metrics import accuracy_score
from algorithms.LinearRegression import LinearRegression
from algorithms.LogisticRegression import LogisticRegression
from algorithms.Perceptron import Perceptron
from algorithms.KMeans import KMeans

# =========================
# PRUEBA REGRESIÓN LINEAR CON UNA VARIABLE
# =========================

def linearRegression (X, y):
    # Dataset sintético, garantiza que todos los numeros random que se generen sigan la misma secuencia
    np.random.seed(42)
    gd_model = LinearRegression(learning_rate=0.05, n_iterations=1000, early_stopping=True)
    gd_model.fit(X, y)

    # ---- Modelo SKL ----
    sk_model = SklearnLR()
    sk_model.fit(X, y)

    # pesos, bias y coeficiente de determinación
    print("[GD]  w, b, R2:", gd_model.weights, gd_model.bias, round(gd_model.score(X, y), 5))
    # pesos, bias y coeficiente de determinación
    print("[SKL] w, b, R2:", sk_model.coef_, sk_model.intercept_, round(sk_model.score(X, y), 5))
    # muestra la diferencia entre los resultados de ambos algoritmos
    print("\n|GD vs SKL|  w:", np.abs(gd_model.weights - sk_model.coef_), 
      " b:", abs(gd_model.bias - sk_model.intercept_))
    
#rand() genera una matriz de 100 filas y 1 columna con números random del 0 al 1 (sin incluirlo)
#X = 2 * np.random.rand(100, 1) # *2 para que este en el rango de 0 a 2 (sin incluirlo)
#toma todas las filas de la columna 0 en X (relación lineal)
#y = 3 * X[:, 0] + 5 + np.random.randn(100) * 0.5  # y = 3x + 5 + ruido (representa la dispersión de los puntos al rededor de la recta)
#linearRegression(X, y)


# =========================
# PRUEBA REGRESIÓN LINEAR CON VARIABLES MÚLTIPLES
# =========================

def multiLinearRegression(X_multi, y_multi):
    np.random.seed(42)

    # mi modelo
    gd_model_multi = LinearRegression(learning_rate=0.05, n_iterations=3000, early_stopping=True)
    gd_model_multi.fit(X_multi, y_multi)
    y_pred_gd_multi = gd_model_multi.predict(X_multi)

    # ---- Modelo SKL ----
    sk_model_multi = SklearnLR()
    sk_model_multi.fit(X_multi, y_multi)
    y_pred_skl_multi = sk_model_multi.predict(X_multi)

    # Resultados comparados
    print("\n--- PRUEBA MULTIVARIABLE ---")
    print("[GD]  [w1, W2], b, R2:", gd_model_multi.weights, gd_model_multi.bias, round(gd_model_multi.score(X_multi, y_multi), 5))
    print("[SKL] w, b, R2:", sk_model_multi.coef_, sk_model_multi.intercept_, round(sk_model_multi.score(X_multi, y_multi), 5))
    print("\n|GD vs SKL|  w:", np.abs(gd_model_multi.weights - sk_model_multi.coef_), 
        " b:", abs(gd_model_multi.bias - sk_model_multi.intercept_))
    
# Generamos X con 4 variables independientes (4 columnas)
#X_multi = 2 * np.random.rand(100, 4)  # 100 muestras, 4 features
# Relación lineal: y = 4*x1 + 2*x2 + 7 + ruido
#y_multi = 4 * X_multi[:, 0] + 10 * X_multi[:, 1] + 7 * X_multi[:, 2] + 3 * X_multi[:, 3] + 7 + np.random.randn(100) * 0.5
#multiLinearRegression(X_multi, y_multi)



# =========================
# PRUEBAS REGRESIÓN LOGÍSTICA
# =========================

def logisticRegression(X_log, y_log): 
    # Dataset binario sintético (linealmente separable)
    np.random.seed(42)

    # ---- Tu modelo GD ----
    gd_model_log = LogisticRegression(learning_rate=0.1, n_iterations=5000)
    gd_model_log.fit(X_log, y_log)
    y_pred_gd_log = gd_model_log.predict(X_log)

    # ---- Modelo SKL ----
    sk_model_log = SklearnLogR()
    sk_model_log.fit(X_log, y_log)
    y_pred_skl_log = sk_model_log.predict(X_log)

    # Resultados comparados
    print("\n--- PRUEBA REGRESIÓN LOGÍSTICA ---")
    print("[GD]  Pesos:", gd_model_log.weights, " Bias:", gd_model_log.bias, " Accuracy:", accuracy_score(y_log, y_pred_gd_log))
    print("[SKL] Pesos:", sk_model_log.coef_, " Bias:", sk_model_log.intercept_, " Accuracy:", accuracy_score(y_log, y_pred_skl_log))
    print("\n|GD vs SKL|  w:", np.abs(gd_model_log.weights - sk_model_log.coef_), 
        " b:", abs(gd_model_log.bias - sk_model_log.intercept_))
    
X_log = 2 * np.random.rand(100, 2) - 1  # valores entre -1 y 1
# regla: si x1 + x2 > 0 => clase 1, else clase 0 (con algo de ruido)
y_log = (X_log[:, 0] + X_log[:, 1] + np.random.randn(100) * 0.2 > 0).astype(int)
logisticRegression(X_log, y_log)

    
# =========================
# PRUEBAS PERCEPTRON
# =========================
def prueba_perceptron_binario():
    np.random.seed(0)
    # Datos 2D separables con ruido leve
    n = 200
    X_pos = np.random.randn(n//2, 2) + np.array([2.0, 2.0])
    X_neg = np.random.randn(n//2, 2) + np.array([-2.0, -2.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*(n//2) + [0]*(n//2))   # etiquetas {0,1}

    pct = Perceptron(learning_rate=1.0, n_iterations=1000, early_stopping=True,
                     patience=10, verbose=True, random_state=42)
    pct.fit(X, y)
    acc = pct.score(X, y)
    print("[Perceptrón binario] accuracy:", round(acc, 4))
    print("Errores por época:", pct.errors_history_[:10], " ...")

def prueba_perceptron_multiclase():
    np.random.seed(1)
    # Tres nubes (3 clases)
    n = 300
    X1 = np.random.randn(n//3, 2) + np.array([2.5, 2.5])
    X2 = np.random.randn(n//3, 2) + np.array([-2.5, 2.5])
    X3 = np.random.randn(n//3, 2) + np.array([0.0, -2.5])
    X = np.vstack([X1, X2, X3])
    y = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))  # 3 clases

    pct = Perceptron(learning_rate=1.0, n_iterations=1000, early_stopping=True,
                     patience=10, verbose=True, random_state=7)
    pct.fit(X, y)
    acc = pct.score(X, y)
    print("[Perceptrón multiclase OVR] accuracy:", round(acc, 4))
    print("Errores totales por época:", pct.errors_history_[:10], " ...")

#prueba_perceptron_binario()
#prueba_perceptron_multiclase()


def _purity_score(y_true, y_pred):
    """Medida simple de pureza en datos sintéticos: por cluster,
    toma la clase mayoritaria y promedia."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = y_true.size
    correct = 0
    for c in np.unique(y_pred):
        mask = (y_pred == c)
        if mask.sum() == 0: 
            continue
        vals, counts = np.unique(y_true[mask], return_counts=True)
        correct += counts.max()
    return correct / total

def prueba_kmeans_basico():
    rng = np.random.default_rng(0)
    n = 300
    C = np.array([[ 2.5,  2.5],
                  [-2.5,  2.5],
                  [ 0.0, -2.5]])
    X = np.vstack([
        rng.normal(C[0], 0.4, size=(n//3, 2)),
        rng.normal(C[1], 0.4, size=(n//3, 2)),
        rng.normal(C[2], 0.4, size=(n//3, 2)),
    ])
    y_true = np.array([0]*(n//3) + [1]*(n//3) + [2]*(n//3))
    idx = rng.permutation(n)  # mezclar
    X, y_true = X[idx], y_true[idx]

    km = KMeans(n_clusters=3, init='k-means++', n_init=5, max_iter=100,
                tol=1e-4, random_state=42, verbose=True)
    y_pred = km.fit_predict(X)
    print("[KMEANS] inertia_final:", round(km.inertia_, 4), "  n_iter:", km.n_iter_)
    print("[KMEANS] centers:\n", km.cluster_centers_)
    print("[KMEANS] inertia_history (10 primeras):", [round(v, 4) for v in km.inertia_history_[:10]])
    print("[KMEANS] purity:", round(_purity_score(y_true, y_pred), 4))

def prueba_kmeans_inits():
    rng = np.random.default_rng(1)
    n = 450
    C = np.array([[ 2.5,  2.5],
                  [-2.5,  2.5],
                  [ 0.0, -2.5]])
    X = np.vstack([
        rng.normal(C[0], 0.6, size=(n//3, 2)),
        rng.normal(C[1], 0.6, size=(n//3, 2)),
        rng.normal(C[2], 0.6, size=(n//3, 2)),
    ])

    km_pp = KMeans(n_clusters=3, init='k-means++', n_init=1, random_state=0).fit(X)
    km_rd = KMeans(n_clusters=3, init='random',   n_init=1, random_state=0).fit(X)
    print("[KMEANS] inertia k-means++:", round(km_pp.inertia_, 2),
          "  inertia random:", round(km_rd.inertia_, 2))


# =========================
# PRUEBAS KMeans
# =========================
#prueba_kmeans_basico()
#prueba_kmeans_inits()