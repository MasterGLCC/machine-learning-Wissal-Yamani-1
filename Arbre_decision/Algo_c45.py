# importer numpy pour les calculs mathématiques
import numpy as np

# importer counter pour compter les classes
from collections import Counter


# fonction pour calculer l'entropie
def entropy(y):

    # compter les occurrences des classes
    counts = Counter(y)

    # récupérer la taille totale
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

    # récupérer les valeurs uniques
    values = set(X_column)

    # initialiser l'entropie pondérée
    weighted_entropy = 0

    # parcourir chaque valeur
    for v in values:

        # récupérer les indices correspondants
        indices = [i for i in range(len(X_column)) if X_column[i] == v]

        # créer le sous-ensemble des classes
        y_subset = [y[i] for i in indices]

        # calculer le poids
        weight = len(y_subset) / len(y)

        # ajouter l'entropie pondérée
        weighted_entropy += weight * entropy(y_subset)

    # retourner le gain d'information
    return H_parent - weighted_entropy


# fonction pour calculer le split info
def split_info(X_column):

    # récupérer la taille totale
    total = len(X_column)

    # récupérer les valeurs uniques
    values = set(X_column)

    # initialiser split info
    SI = 0

    # parcourir chaque valeur
    for v in values:

        # calculer la probabilité
        p = X_column.count(v) / total

        # appliquer la formule
        SI -= p * np.log2(p)

    # retourner split info
    return SI


# fonction pour calculer le gain ratio
def gain_ratio(X_column, y):

    # calculer information gain
    IG = information_gain(X_column, y)

    # calculer split info
    SI = split_info(X_column)

    # éviter division par zéro
    if SI == 0:
        return 0

    # retourner gain ratio
    return IG / SI


# fonction pour trouver le meilleur seuil
def best_threshold(feature_values, y):

    # trier les données
    sorted_data = sorted(zip(feature_values, y))

    # initialiser meilleur gain
    best_gain = -1

    # initialiser meilleur seuil
    best_t = None

    # parcourir les valeurs
    for i in range(len(sorted_data) - 1):

        # calculer un seuil candidat
        t = (sorted_data[i][0] + sorted_data[i+1][0]) / 2

        # créer un split binaire
        X_split = ["<=t" if x <= t else ">t" for x in feature_values]

        # calculer le gain ratio
        gain = gain_ratio(X_split, y)

        # vérifier si le gain est meilleur
        if gain > best_gain:

            # mettre à jour le gain
            best_gain = gain

            # mettre à jour le seuil
            best_t = t

    # retourner le meilleur seuil
    return best_t


# fonction pour trouver la meilleure feature
def best_feature_c45(X, y):

    # dictionnaire des scores
    scores = {}

    # parcourir les features
    for feature in X:

        # calculer gain ratio
        scores[feature] = gain_ratio(X[feature], y)

    # retourner la feature avec meilleur score
    return max(scores, key=scores.get)


# fonction récursive de construction de l'arbre
def build_tree_c45(X, y, features):

    # vérifier si toutes les classes sont identiques
    if len(set(y)) == 1:

        # retourner la classe
        return y[0]

    # vérifier s'il reste des features
    if not features:

        # retourner la classe majoritaire
        return Counter(y).most_common(1)[0][0]

    # choisir la meilleure feature
    best = best_feature_c45(X, y)

    # créer le noeud de l'arbre
    tree = {best: {}}

    # récupérer les valeurs possibles
    values = set(X[best])

    # parcourir les valeurs
    for v in values:

        # récupérer les indices correspondants
        indices = [i for i in range(len(y)) if X[best][i] == v]

        # créer le sous-dataset
        sub_X = {
            f: [X[f][i] for i in indices]
            for f in features if f != best
        }

        # créer les sous-classes
        sub_y = [y[i] for i in indices]

        # construire récursivement le sous-arbre
        tree[best][v] = build_tree_c45(
            sub_X,
            sub_y,
            [f for f in features if f != best]
        )

    # retourner l'arbre
    return tree


# fonction de pruning
def prune_tree(tree, X_val, y_val):

    # vérifier si on est sur une feuille
    if not isinstance(tree, dict):

        # retourner la feuille
        return tree

    # récupérer la feature
    feature = list(tree.keys())[0]

    # parcourir les branches
    for value in tree[feature]:

        # appliquer récursivement le pruning
        tree[feature][value] = prune_tree(
            tree[feature][value],
            X_val,
            y_val
        )

    # calculer précision actuelle
    current_acc = evaluate(tree, X_val, y_val)

    # récupérer classe majoritaire
    majority = Counter(y_val).most_common(1)[0][0]

    # calculer précision simplifiée
    baseline_acc = evaluate(majority, X_val, y_val)

    # vérifier si pruning améliore le modèle
    if baseline_acc >= current_acc:

        # couper le sous-arbre
        return majority

    # retourner arbre final
    return tree


# dataset d'entraînement
X = {

    # feature age
    "age": [20, 22, 25, 35, 40, 45],

    # feature salaire
    "salaire": [
        "faible",
        "faible",
        "moyen",
        "eleve",
        "eleve",
        "eleve"
    ]
}

# classes à prédire
y = ["non", "non", "non", "oui", "oui", "oui"]


# récupérer la liste des features
features = list(X.keys())


# construire l'arbre c4.5
tree = build_tree_c45(X, y, features)


# afficher l'arbre
print("\narbre c4.5 :\n")

print(tree)


# fonction de prédiction
def predict(tree, sample):

    # vérifier si on est sur une feuille
    if not isinstance(tree, dict):

        # retourner la prédiction
        return tree

    # récupérer la feature racine
    feature = list(tree.keys())[0]

    # récupérer la valeur de l'exemple
    value = sample[feature]

    # suivre la branche correspondante
    subtree = tree[feature][value]

    # continuer récursivement
    return predict(subtree, sample)


# fonction d'évaluation
def evaluate(tree, X_test, y_test):

    # initialiser le nombre correct
    correct = 0

    # parcourir les données
    for i in range(len(y_test)):

        # créer un exemple
        sample = {
            feature: X_test[feature][i]
            for feature in X_test
        }

        # faire la prédiction
        pred = predict(tree, sample)

        # vérifier si la prédiction est correcte
        if pred == y_test[i]:

            # augmenter le compteur
            correct += 1

    # retourner l'accuracy
    return correct / len(y_test)


# nouvel exemple à prédire
new_sample = {

    "age": 25,
    "salaire": "moyen"
}


# faire la prédiction
result = predict(tree, new_sample)


# afficher le résultat
print("\nprédiction :", result)