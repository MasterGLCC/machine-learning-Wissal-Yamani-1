# importer numpy pour les calculs mathématiques
import numpy as np

# importer counter pour compter les classes
from collections import Counter


# fonction pour calculer l'entropie
def entropy(y):

    # compter le nombre d'exemples de chaque classe
    counts = Counter(y)

    # récupérer la taille totale du dataset
    total = len(y)

    # initialiser l'entropie
    H = 0

    # parcourir chaque classe
    for count in counts.values():

        # calculer la probabilité
        p = count / total

        # appliquer la formule de l'entropie
        H -= p * np.log2(p)

    # retourner l'entropie
    return H


# fonction pour calculer le gain d'information
def information_gain(X_column, y):

    # calculer l'entropie initiale
    H_parent = entropy(y)

    # récupérer les valeurs uniques de la feature
    values = set(X_column)

    # initialiser l'entropie pondérée
    weighted_entropy = 0

    # parcourir les valeurs possibles
    for v in values:

        # récupérer les indices correspondants
        indices = [
            i for i in range(len(X_column))
            if X_column[i] == v
        ]

        # créer le sous-ensemble des classes
        y_subset = [y[i] for i in indices]

        # calculer le poids du sous-ensemble
        weight = len(y_subset) / len(y)

        # ajouter l'entropie pondérée
        weighted_entropy += weight * entropy(y_subset)

    # calculer le gain d'information
    gain = H_parent - weighted_entropy

    # retourner le gain
    return gain


# fonction pour trouver la meilleure feature
def best_feature(X, y):

    # créer un dictionnaire des gains
    gains = {}

    # parcourir chaque feature
    for feature in X:

        # calculer le gain de la feature
        gains[feature] = information_gain(
            X[feature],
            y
        )

    # retourner la feature avec le gain maximal
    return max(gains, key=gains.get)


# fonction récursive pour construire l'arbre id3
def build_tree(X, y, features):

    # vérifier si toutes les classes sont identiques
    if len(set(y)) == 1:

        # retourner la classe
        return y[0]

    # vérifier s'il reste des features
    if not features:

        # retourner la classe majoritaire
        return Counter(y).most_common(1)[0][0]

    # choisir la meilleure feature
    best = best_feature(X, y)

    # créer le noeud racine
    tree = {best: {}}

    # récupérer les valeurs possibles
    values = set(X[best])

    # parcourir les valeurs
    for v in values:

        # récupérer les indices correspondants
        indices = [
            i for i in range(len(y))
            if X[best][i] == v
        ]

        # créer le sous-dataset
        sub_X = {

            f: [X[f][i] for i in indices]

            for f in features
            if f != best
        }

        # créer les sous-classes
        sub_y = [y[i] for i in indices]

        # construire récursivement le sous-arbre
        subtree = build_tree(
            sub_X,
            sub_y,
            [f for f in features if f != best]
        )

        # ajouter la branche dans l'arbre
        tree[best][v] = subtree

    # retourner l'arbre final
    return tree


# dataset d'entraînement
X = {

    # feature age
    "age": ["jeune", "jeune", "vieux", "vieux"],

    # feature salaire
    "salaire": ["faible", "eleve", "faible", "eleve"]
}

# classes à prédire
y = ["non", "oui", "oui", "oui"]


# récupérer la liste des features
features = list(X.keys())


# construire l'arbre id3
tree = build_tree(X, y, features)


# afficher l'arbre
print(tree)


# fonction de prédiction
def predict(tree, sample):

    # récupérer la racine de l'arbre
    root = list(tree.keys())[0]

    # récupérer la valeur de la feature
    value = sample[root]

    # suivre la branche correspondante
    subtree = tree[root][value]

    # vérifier si on est sur une feuille
    if not isinstance(subtree, dict):

        # retourner la prédiction
        return subtree

    # continuer récursivement
    return predict(subtree, sample)


# nouvel exemple à classifier
new_sample = {

    "age": "jeune",
    "salaire": "faible"
}


# faire la prédiction
result = predict(tree, new_sample)


# afficher le résultat
print("\nprédiction :", result)