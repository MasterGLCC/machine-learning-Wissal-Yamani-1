# importer numpy pour les calculs mathématiques
import numpy as np

# importer counter pour compter les classes
from collections import Counter


# fonction pour calculer l'indice de gini
def gini(y):

    # compter les occurrences des classes
    counts = Counter(y)

    # récupérer le nombre total d'exemples
    total = len(y)

    # initialiser l'impureté
    impurity = 0

    # parcourir chaque classe
    for count in counts.values():

        # calculer la probabilité
        p = count / total

        # ajouter le carré de la probabilité
        impurity += p ** 2

    # retourner l'indice de gini
    return 1 - impurity


# fonction pour effectuer un split binaire
def split(X_column, y, threshold):

    # créer le groupe gauche
    left_y = [
        y[i]
        for i in range(len(X_column))
        if X_column[i] <= threshold
    ]

    # créer le groupe droit
    right_y = [
        y[i]
        for i in range(len(X_column))
        if X_column[i] > threshold
    ]

    # retourner les deux groupes
    return left_y, right_y


# fonction pour calculer le gini pondéré d'un split
def gini_split(X_column, y, threshold):

    # diviser les données
    left_y, right_y = split(X_column, y, threshold)

    # récupérer la taille totale
    n = len(y)

    # calculer le gini du groupe gauche
    g_left = gini(left_y)

    # calculer le gini du groupe droit
    g_right = gini(right_y)

    # calculer la moyenne pondérée
    score = (
        (len(left_y) / n) * g_left
        + (len(right_y) / n) * g_right
    )

    # retourner le score
    return score


# fonction pour trouver le meilleur seuil
def best_split(X_column, y):

    # initialiser le meilleur seuil
    best_threshold = None

    # initialiser le meilleur score
    best_score = float("inf")

    # récupérer les valeurs uniques triées
    unique_values = sorted(set(X_column))

    # parcourir les valeurs
    for i in range(len(unique_values) - 1):

        # calculer un seuil candidat
        threshold = (
            unique_values[i]
            + unique_values[i + 1]
        ) / 2

        # calculer le score du split
        score = gini_split(X_column, y, threshold)

        # vérifier si le score est meilleur
        if score < best_score:

            # mettre à jour le score
            best_score = score

            # mettre à jour le seuil
            best_threshold = threshold

    # retourner le meilleur seuil
    return best_threshold


# fonction récursive pour construire l'arbre cart
def build_cart(X, y, features):

    # vérifier si toutes les classes sont identiques
    if len(set(y)) == 1:

        # retourner la classe
        return y[0]

    # vérifier s'il reste des features
    if not features:

        # retourner la classe majoritaire
        return Counter(y).most_common(1)[0][0]

    # initialiser la meilleure feature
    best_feature = None

    # initialiser le meilleur seuil
    best_threshold = None

    # initialiser le meilleur score
    best_score = float("inf")

    # parcourir les features
    for f in features:

        # trouver le meilleur seuil
        threshold = best_split(X[f], y)

        # vérifier si un seuil existe
        if threshold is None:
            continue

        # calculer le score du split
        score = gini_split(X[f], y, threshold)

        # vérifier si le split est meilleur
        if score < best_score:

            # mettre à jour le score
            best_score = score

            # mettre à jour la feature
            best_feature = f

            # mettre à jour le seuil
            best_threshold = threshold

    # vérifier si aucun split valide
    if best_feature is None:

        # retourner la classe majoritaire
        return Counter(y).most_common(1)[0][0]

    # créer le noeud de l'arbre
    tree = {f"{best_feature} <= {best_threshold}": {}}

    # récupérer les indices du groupe gauche
    left_idx = [
        i
        for i in range(len(y))
        if X[best_feature][i] <= best_threshold
    ]

    # récupérer les indices du groupe droit
    right_idx = [
        i
        for i in range(len(y))
        if X[best_feature][i] > best_threshold
    ]

    # construire le sous-arbre gauche
    tree[f"{best_feature} <= {best_threshold}"]["left"] = build_cart(

        # créer les données gauche
        {
            f: [X[f][i] for i in left_idx]
            for f in features
        },

        # créer les classes gauche
        [y[i] for i in left_idx],

        # envoyer les features
        features
    )

    # construire le sous-arbre droit
    tree[f"{best_feature} <= {best_threshold}"]["right"] = build_cart(

        # créer les données droite
        {
            f: [X[f][i] for i in right_idx]
            for f in features
        },

        # créer les classes droite
        [y[i] for i in right_idx],

        # envoyer les features
        features
    )

    # retourner l'arbre final
    return tree


# dataset d'entraînement
X = {

    # feature age
    "age": [20, 22, 25, 35, 40, 45],

    # feature salaire
    "salaire": [2000, 2500, 3000, 5000, 5500, 6000]
}

# classes à prédire
y = ["non", "non", "non", "oui", "oui", "oui"]


# récupérer la liste des features
features = list(X.keys())


# construire l'arbre cart
tree = build_cart(X, y, features)


# afficher l'arbre
print("\narbre cart :\n")

print(tree)


# fonction de prédiction
def predict_cart(tree, sample):

    # vérifier si on est sur une feuille
    if not isinstance(tree, dict):

        # retourner la prédiction
        return tree

    # récupérer le noeud
    node = list(tree.keys())[0]

    # séparer la feature et le seuil
    feature, _, threshold = node.partition(" <= ")

    # convertir le seuil en float
    threshold = float(threshold)

    # vérifier la branche à suivre
    if sample[feature] <= threshold:

        # aller à gauche
        subtree = tree[node]["left"]

    else:

        # aller à droite
        subtree = tree[node]["right"]

    # continuer récursivement
    return predict_cart(subtree, sample)


# nouvel exemple à classifier
new_sample = {

    "age": 30,
    "salaire": 4000
}


# faire la prédiction
result = predict_cart(tree, new_sample)


# afficher le résultat
print("\nprédiction :", result)

