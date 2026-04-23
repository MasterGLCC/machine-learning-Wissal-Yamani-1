import numpy as np  # calculs numériques
import pandas as pd  # manipulation des données

from sklearn.linear_model import LinearRegression  
# modèle de régression linéaire

from sklearn.preprocessing import PolynomialFeatures  
# permet de créer x², x³, etc.


# 1. LOAD DATA
data = pd.read_csv("../Datasets/polynomial_data.csv")
# lecture du fichier CSV

X = data[["x"]].values  
# IMPORTANT : doit être 2D (matrice), même pour une seule variable

y = data["y"].values  
# variable cible

# 2. TRANSFORMATION POLYNOMIALE
poly = PolynomialFeatures(degree=2)  
# on veut x et x²

X_poly = poly.fit_transform(X)  
# transforme :
# x → [1, x, x²]


# 3. MODEL
model = LinearRegression()  
# modèle linéaire classique


# 4. TRAINING
model.fit(X_poly, y)  
# apprentissage sur les données transformées


# 5. RESULTATS
print("Coefficients:", model.coef_)  
# poids pour [1, x, x²]

print("Intercept:", model.intercept_)  
# biais


# 6. TEST
x_test = np.array([[12]])
# valeur test

x_test_poly = poly.transform(x_test)  
# transforme en [1, x, x²]

prediction = model.predict(x_test_poly)  
# prédiction

print("\nPrediction for x=12:", prediction[0])