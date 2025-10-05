import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Importar tus modelos
from algorithms.LinearRegression import LinearRegression
from algorithms.LogisticRegression import LogisticRegression
from algorithms.DecisionTreeClassifier import DecisionTreeClassifier
from algorithms.Perceptron import Perceptron
from algorithms.NaiveBayes import NaiveBayes
from algorithms.MLP import MLPClassifier
from algorithms.PCA import PCA
from algorithms.KMeans import KMeans

# ========= DATASETS =========
# Regresión (y continua)
X_reg, y_reg = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Clasificación (y binario/multiclase)
X_clf, y_clf = make_classification(n_samples=200, n_features=5, n_classes=2, n_informative=3, random_state=42)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

print("\n=== REGRESIÓN LINEAL ===")
lr = LinearRegression(learning_rate=0.01, n_iterations=1000)
lr.fit(Xr_train, yr_train)
print("R² en test:", lr.score(Xr_test, yr_test))
print("Real vs Predicho (primeros 5):")
print(np.c_[yr_test[:5], lr.predict(Xr_test[:5])])

print("\n=== REGRESIÓN LOGÍSTICA ===")
logr = LogisticRegression(learning_rate=0.1, n_iterations=2000)
logr.fit(Xc_train, yc_train)
print("Accuracy:", logr.score(Xc_test, yc_test))
print("Real vs Predicho (primeros 10):")
print(np.c_[yc_test[:10], logr.predict(Xc_test[:10])])

print("\n=== ÁRBOL DE DECISIÓN ===")
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(Xc_train, yc_train)
print("Accuracy:", tree.score(Xc_test, yc_test))
print("Real vs Predicho (primeros 10):")
print(np.c_[yc_test[:10], tree.predict(Xc_test[:10])])

print("\n=== PERCEPTRÓN ===")
pct = Perceptron(n_iterations=50, learning_rate=0.1)
pct.fit(Xc_train, yc_train)
print("Accuracy:", pct.score(Xc_test, yc_test))
print("Real vs Predicho (primeros 10):")
print(np.c_[yc_test[:10], pct.predict(Xc_test[:10])])

print("\n=== NAIVE BAYES (Gaussiano) ===")
nb = NaiveBayes(nb_type='gaussian')
nb.fit(Xc_train, yc_train)
print("Accuracy:", nb.score(Xc_test, yc_test))
print("Real vs Predicho (primeros 10):")
print(np.c_[yc_test[:10], nb.predict(Xc_test[:10])])

print("\n=== MLP (Red Neuronal) ===")
mlp = MLPClassifier(hidden_layers=(10,5), n_iterations=200, learning_rate=0.01)
mlp.fit(Xc_train, yc_train)
print("Accuracy:", mlp.score(Xc_test, yc_test))
print("Real vs Predicho (primeros 10):")
print(np.c_[yc_test[:10], mlp.predict(Xc_test[:10])])

print("\n=== PCA ===")
pca = PCA(n_components=2)
Xc_proj = pca.fit_transform(X_clf)
print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
print("Proyección (primeros 5 puntos):")
print(Xc_proj[:5])

print("\n=== KMEANS ===")
km = KMeans(n_clusters=2, random_state=42)
km.fit(X_clf)
print("Inercia (menor = mejor):", km.inertia_)
print("Etiquetas asignadas (primeros 10):", km.labels_[:10])
