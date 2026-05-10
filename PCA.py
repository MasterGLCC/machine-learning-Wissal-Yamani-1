import numpy as np  # bibliothèque pour les calculs numériques


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components  # nombre de composantes principales

    def fit(self, X):

        # calcul de la moyenne de chaque feature
        self.mean = np.mean(X, axis=0)

        # centrage des données
        X_centered = X - self.mean

        # calcul matrice covariance
        cov_matrix = np.cov(X_centered.T)

        # calcul valeurs propres et vecteurs propres
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # tri des valeurs propres du plus grand au plus petit
        idxs = np.argsort(eigenvalues)[::-1]

        # réorganisation des valeurs propres
        eigenvalues = eigenvalues[idxs]

        # réorganisation des vecteurs propres
        eigenvectors = eigenvectors[:, idxs]

        # sélectionner les meilleures composantes
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):

        # centrage des données
        X_centered = X - self.mean

        # projection sur les composantes principales
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):

        # entraînement PCA
        self.fit(X)

        # transformation des données
        return self.transform(X)


# dataset simple
X = np.array([
    [170, 65],
    [175, 70],
    [180, 75],
    [185, 80]
])

# création modèle PCA
pca = PCA(n_components=1)

# réduction dimension
X_reduced = pca.fit_transform(X)

# affichage données réduites
print("données réduites :\n")

print(X_reduced)