import numpy as np
from collections import defaultdict, Counter


class NaiveBayes:

    def fit(self, X, y):

        self.classes = np.unique(y)  # classes possibles
        self.class_priors = {}  # P(C)
        self.feature_probs = {}  # P(X|C)

        n_samples = len(y)

        # calcul des probabilités par classe
        for c in self.classes:

            X_c = X[y == c]  # données de la classe c

            self.class_priors[c] = len(X_c) / n_samples  # P(C)

            self.feature_probs[c] = []

            # pour chaque feature
            for i in range(X.shape[1]):

                feature_values = X_c[:, i]

                counts = Counter(feature_values)

                # stockage des probabilités conditionnelles
                probs = {}

                total = len(feature_values)

                for val, count in counts.items():
                    probs[val] = count / total  # P(Xi | C)

                self.feature_probs[c].append(probs)

    def predict(self, X):

        predictions = []

        for x in X:

            class_scores = {}

            # calcul P(C|X)
            for c in self.classes:

                score = np.log(self.class_priors[c])  # log P(C)

                for i, value in enumerate(x):

                    probs = self.feature_probs[c][i]

                    # si valeur inconnue → petite probabilité
                    p = probs.get(value, 1e-6)

                    score += np.log(p)  # log P(X|C)

                class_scores[c] = score

            # choisir classe avec plus grande probabilité
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)



# dataset simple
X = np.array([
    ["jeune", "faible"],
    ["jeune", "eleve"],
    ["vieux", "faible"],
    ["vieux", "eleve"]
])

y = np.array(["non", "non", "oui", "oui"])

# créer modèle
model = NaiveBayes()

# entraînement
model.fit(X, y)

# nouvelle donnée
X_test = np.array([
    ["jeune", "faible"],
    ["vieux", "eleve"]
])

# prédiction
pred = model.predict(X_test)

print("predictions:", pred)