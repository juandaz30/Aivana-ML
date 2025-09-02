import numpy as np
from sklearn.decomposition import PCA as SKPCA
from algorithms.PCA import PCA


# --- Dataset sintético ---
rng = np.random.default_rng(42)
X = rng.normal(size=(100, 5))  # 100 muestras, 5 features

# --- Caso 1: mismos n_components (int) ---
my_pca = PCA(n_components=2).fit(X)
sk_pca = SKPCA(n_components=2).fit(X)

print("\n--- PCA int(2) ---")
print("[MY ] Explained variance ratio:", my_pca.explained_variance_ratio_)
print("[SKL] Explained variance ratio:", sk_pca.explained_variance_ratio_)
print("ratio:", np.abs(my_pca.explained_variance_ratio_ - sk_pca.explained_variance_ratio_))

# comparando proyecciones
Z_my = my_pca.transform(X)
Z_sk = sk_pca.transform(X)
print("Projections corr (col1, col2):", 
      np.corrcoef(Z_my[:,0], Z_sk[:,0])[0,1],
      np.corrcoef(Z_my[:,1], Z_sk[:,1])[0,1])

# --- Caso 2: proporción de varianza ---
my_pca95 = PCA(n_components=0.95).fit(X)
sk_pca95 = SKPCA(n_components=0.95).fit(X)

print("\n--- PCA varianza 95% ---")
print("[MY ] n_components_:", my_pca95.n_components_)
print("[SKL] n_components_:", sk_pca95.n_components_)

# --- Caso 3: Whitening ---
my_white = PCA(n_components=3, whiten=True).fit_transform(X)
sk_white = SKPCA(n_components=3, whiten=True).fit_transform(X)

print("\n--- PCA whitening ---")
print("[MY ] var(Z):", my_white.var(axis=0))
print("[SKL] var(Z):", sk_white.var(axis=0))