# importer numpy pour les calculs numériques
import numpy as np


# classe cart simplifiée (arbre de régression)
class CART:

    # entraînement du modèle
    def fit(self, X, y):

        # on choisit une seule feature (simplification)
        self.feature_index = 0

        # on calcule le seuil comme la moyenne
        self.threshold = np.mean(X[:, self.feature_index])

        # masque pour la partie gauche
        left_mask = X[:, self.feature_index] <= self.threshold

        # masque pour la partie droite
        right_mask = X[:, self.feature_index] > self.threshold

        # valeur prédite à gauche (moyenne des labels)
        self.left_value = np.mean(y[left_mask])

        # valeur prédite à droite (moyenne des labels)
        self.right_value = np.mean(y[right_mask])


    # prédiction
    def predict(self, X):

        # tableau de prédictions
        preds = np.zeros(len(X))

        # parcourir les données
        for i in range(len(X)):

            # si valeur <= seuil
            if X[i, self.feature_index] <= self.threshold:

                # prédire valeur gauche
                preds[i] = self.left_value

            else:

                # prédire valeur droite
                preds[i] = self.right_value

        # retourner prédictions
        return preds


# fonction erreur quadratique moyenne
def mse(y_true, y_pred):

    # calculer la moyenne des erreurs au carré
    return np.mean((y_true - y_pred) ** 2)


# fonction gradient (résidus)
def gradient(y, y_pred):

    # erreur entre vraie valeur et prédiction
    return y - y_pred


# modèle simple de boosting
class SimpleXGBoost:

    # constructeur
    def __init__(self, n_estimators=5, learning_rate=0.1):

        # nombre d'arbres
        self.n_estimators = n_estimators

        # taux d'apprentissage
        self.learning_rate = learning_rate

        # liste des modèles
        self.models = []


    # entraînement
    def fit(self, X, y):

        # prédiction initiale à zéro
        self.pred = np.zeros(len(y))

        # boucle de boosting
        for i in range(self.n_estimators):

            # calcul des erreurs (gradient)
            grad = gradient(y, self.pred)

            # création d'un arbre cart
            tree = CART()

            # entraînement sur les erreurs
            tree.fit(X, grad)

            # stocker l'arbre
            self.models.append(tree)

            # prédiction de l'arbre
            update = tree.predict(X)

            # mise à jour du modèle global
            self.pred += self.learning_rate * update

            # affichage de la loss
            print(f"Iteration {i}, Loss: {mse(y, self.pred)}")


    # prédiction finale
    def predict(self, X):

        # initialisation
        pred = np.zeros(len(X))

        # somme des arbres
        for tree in self.models:

            # ajouter contribution de chaque arbre
            pred += self.learning_rate * tree.predict(X)

        # retourner résultat final
        return pred


# dataset simple
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])

# modèle xgboost simplifié
model = SimpleXGBoost(n_estimators=50, learning_rate=0.1)

# entraînement du modèle
model.fit(X, y)

# prédiction
predictions = model.predict(X)

# afficher résultats
print("Predictions:", predictions)