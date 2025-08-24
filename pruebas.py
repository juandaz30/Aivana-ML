import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR

# Dataset sintético, garantiza que todos los numeros random que se generen sigan la misma secuencia
np.random.seed(42)

# =========================
# PRUEBA CON UNA VARIABLE
# =========================

#rand() genera una matriz de 100 filas y 1 columna con números random del 0 al 1 (sin incluirlo)
# se puede generar un vector de varias dimensiones cambiando el 1
X = 2 * np.random.rand(100, 1) # *2 para que este en el rango de 0 a 2 (sin incluirlo)
#toma todas las filas de la columna 0 en X (relación lineal)
y = 3 * X[:, 0] + 5 + np.random.randn(100) * 0.5  # y = 3x + 5 + ruido (representa la dispersión de los puntos al rededor de la recta)

# ---- Tu modelo GD ----
from algorithms.LinearRegression import LinearRegression  # <-- importa tu implementación

gd_model = LinearRegression(learning_rate=0.05, n_iterations=2000, early_stopping=True)
gd_model.fit(X, y)
y_pred_gd = gd_model.predict(X)

# ---- Modelo SKL ----
sk_model = SklearnLR()
sk_model.fit(X, y)
y_pred_skl = sk_model.predict(X)

# pesos, bias y coeficiente de determinación
print("[GD]  w, b, R2:", gd_model.weights, gd_model.bias, round(gd_model.score(X, y), 5))
# pesos, bias y coeficiente de determinación
print("[SKL] w, b, R2:", sk_model.coef_, sk_model.intercept_, round(sk_model.score(X, y), 5))
# muestra la diferencia entre los resultados de ambos algoritmos
print("\nΔ|GD vs SKL|  w:", np.abs(gd_model.weights - sk_model.coef_), 
      " b:", abs(gd_model.bias - sk_model.intercept_))



# =========================
# PRUEBA CON VARIABLES MÚLTIPLES
# =========================
np.random.seed(42)
# Generamos X con 2 variables independientes (2 columnas)
X_multi = 2 * np.random.rand(100, 2)  # 100 muestras, 2 features

# Relación lineal: y = 4*x1 + 2*x2 + 7 + ruido
y_multi = 4 * X_multi[:, 0] + 2 * X_multi[:, 1] + 7 + np.random.randn(100) * 0.5

# ---- Tu modelo GD ----
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
print("\nΔ|GD vs SKL|  w:", np.abs(gd_model_multi.weights - sk_model_multi.coef_), 
      " b:", abs(gd_model_multi.bias - sk_model_multi.intercept_))
