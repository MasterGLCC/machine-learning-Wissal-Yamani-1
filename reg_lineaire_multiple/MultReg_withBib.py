import numpy as np  # bibliothèque pour les calculs numériques (vecteurs, matrices)
import pandas as pd  # bibliothèque pour manipuler les données sous forme de tableaux

from sklearn.linear_model import LinearRegression  
# import du modèle de régression linéaire déjà implémenté dans scikit-learn


# 1. CHARGEMENT DES DONNÉES
data = pd.read_csv("../Datasets/regression_multiple_data.csv")
# lecture du fichier CSV et stockage sous forme de DataFrame (tableau structuré)

X = data[["x1", "x2", "x3"]].values  
# extraction des colonnes x1, x2, x3 comme matrice de features (entrée du modèle)

y = data["y"].values  
# extraction de la colonne y (variable à prédire)


# 2. CRÉATION DU MODÈLE
model = LinearRegression()  
# création d’un objet modèle de régression linéaire
# (le modèle contient déjà l’algorithme interne de calcul des coefficients)


# 3. ENTRAÎNEMENT DU MODÈLE
model.fit(X, y)  
# apprentissage automatique :
# scikit-learn calcule les meilleurs coefficients (w et b)
# en minimisant l’erreur (moindres carrés)


# 4. RÉSULTATS DU MODÈLE
print("Coefficients (w):", model.coef_)  
# affiche les poids appris pour x1, x2, x3

print("Intercept (b):", model.intercept_)  
# affiche le biais (valeur quand toutes les variables = 0)


# 5. TEST DU MODÈLE
x_test = np.array([[3, 7, 6]])  
# nouvel exemple : (étude=3, sommeil=7, motivation=6)

prediction = model.predict(x_test)  
# le modèle calcule automatiquement : y = w·x + b

print("\nPrediction:", prediction[0])  
# affiche le résultat final