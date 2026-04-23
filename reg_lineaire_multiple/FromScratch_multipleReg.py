import numpy as np  # bibliothèque pour les calculs mathématiques (vecteurs, matrices)

# 1. CHARGEMENT DES DONNÉES
data = np.loadtxt("../Datasets/regression_multiple_data.csv", delimiter=",", skiprows=1)
# on charge le fichier CSV (on ignore la première ligne car ce sont les noms des colonnes)

X = data[:, :3]  # on prend les 3 premières colonnes (x1, x2, x3)
y = data[:, 3]   # on prend la dernière colonne (valeur cible y)

n_samples, n_features = X.shape  # nombre d'exemples et nombre de variables


# 2. INITIALISATION DU MODÈLE
w = np.zeros(n_features)  # initialisation des poids (w1, w2, w3) à 0
b = 0.0  # initialisation du biais (intercept)

# 3. HYPERPARAMÈTRES
learning_rate = 0.001  # vitesse d'apprentissage (petit pas à chaque update)
epochs = 5000  # nombre d'itérations pour entraîner le modèle


# 4. DESCENTE DE GRADIENT
for i in range(epochs):  # boucle d'entraînement

    # PRÉDICTION DU MODÈLE
    y_pred = np.dot(X, w) + b
    # produit scalaire entre X et w + biais

    # FONCTION DE COÛT (ERREUR)
    cost = (1 / n_samples) * np.sum((y - y_pred) ** 2)
    # erreur moyenne quadratique (MSE)

    # CALCUL DES GRADIENTS
    dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
    # dérivée de la loss par rapport aux poids

    db = (-2 / n_samples) * np.sum(y - y_pred)
    # dérivée de la loss par rapport au biais

    # MISE À JOUR DES PARAMÈTRES
    w = w - learning_rate * dw  # mise à jour des poids
    b = b - learning_rate * db  # mise à jour du biais

    # AFFICHAGE DE L'APPRENTISSAGE
    if i % 200 == 0:
        print(f"Epoch {i}, Cost: {cost:.4f}")
        # on affiche la perte toutes les 200 itérations


# 5. RÉSULTAT FINAL
print("\nModèle final appris :")
print("Poids (w):", w)  # importance de chaque variable
print("Biais (b):", b)   # valeur de base


# 6. TEST DU MODÈLE
x_test = np.array([3, 7, 6])
# nouvel exemple : étude, sommeil, motivation

prediction = np.dot(x_test, w) + b
# calcul de la prédiction

print("\nPrédiction pour x_test :", prediction)