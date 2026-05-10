import numpy as np  # importation de numpy pour les calculs numériques


# calcul de distance euclidienne
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))  # calcul de la distance entre deux points


class DBSCAN:

    def __init__(self, eps=0.5, min_samples=3):
        self.eps = eps  # distance maximale pour considérer deux points comme voisins
        self.min_samples = min_samples  # nombre minimum de voisins pour être un core point

        self.labels = None  # tableau des labels (-1 = bruit)

    # trouver les voisins d'un point
    def region_query(self, X, point_idx):

        neighbors = []  # liste des voisins

        for i in range(len(X)):  # parcourir tous les points
            if distance(X[point_idx], X[i]) <= self.eps:  # vérifier la distance
                neighbors.append(i)  # ajouter l'indice si voisin

        return neighbors  # retourner les voisins

    # étendre un cluster
    def expand_cluster(self, X, labels, point_idx, cluster_id):

        neighbors = self.region_query(X, point_idx)  # récupérer les voisins

        if len(neighbors) < self.min_samples:  # si pas assez de voisins
            labels[point_idx] = -1  # marquer comme bruit
            return False  # ne pas créer de cluster

        labels[point_idx] = cluster_id  # assigner le cluster au point

        i = 0  # index de parcours des voisins
        while i < len(neighbors):  # parcourir la liste des voisins

            n_idx = neighbors[i]  # récupérer un voisin

            if labels[n_idx] == 0:  # si pas encore visité

                labels[n_idx] = cluster_id  # l'ajouter au cluster

                new_neighbors = self.region_query(X, n_idx)  # nouveaux voisins

                if len(new_neighbors) >= self.min_samples:  # si core point
                    neighbors += new_neighbors  # étendre la recherche

            elif labels[n_idx] == -1:  # si bruit
                labels[n_idx] = cluster_id  # le convertir en cluster

            i += 1  # passer au suivant

        return True  # cluster créé

    # entraînement
    def fit(self, X):

        X = np.array(X)  # convertir en tableau numpy

        n = len(X)  # nombre de points

        labels = np.zeros(n)  # initialisation des labels à 0

        cluster_id = 0  # compteur de clusters

        for i in range(n):  # parcourir tous les points

            if labels[i] != 0:  # si déjà visité
                continue  # passer au suivant

            if self.expand_cluster(X, labels, i, cluster_id + 1):  # essayer créer cluster
                cluster_id += 1  # augmenter id cluster

        self.labels = labels  # sauvegarder les labels
        return labels  # retourner résultat


# prédiction (simple assignation)
def predict(self, X):

    return self.labels  # retourner les labels appris


# dataset simple 2D
X = [
    [1, 2],  # point 1
    [1, 3],  # point 2
    [2, 2],  # point 3
    [8, 8],  # point 4
    [8, 9],  # point 5
    [25, 80]  # bruit
]

# créer modèle
model = DBSCAN(eps=2, min_samples=2)  # initialisation du modèle

# entraîner
labels = model.fit(X)  # apprentissage des clusters
print("clusters :", labels)