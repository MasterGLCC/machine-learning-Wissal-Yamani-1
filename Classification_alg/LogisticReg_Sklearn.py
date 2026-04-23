import numpy as np  # calculs numériques
import pandas as pd  # manipulation des données

from sklearn.linear_model import LogisticRegression  
# modèle de régression logistique déjà implémenté

# 1. LOAD DATA
data = pd.read_csv("../Datasets/Classification_data.csv")
# lecture du fichier CSV

X = data[["hours"]].values  
# variable d'entrée (2D obligatoire pour sklearn)

y = data["label"].values
# variable cible (0 ou 1)


# 2. MODEL
model = LogisticRegression()
# création du modèle

# 3. TRAINING
model.fit(X, y)
# apprentissage automatique (optimisation interne)

# 4. RÉSULTATS
print("Coefficient (w):", model.coef_)  
# poids appris

print("Intercept (b):", model.intercept_)  
# biais

# 5. TEST
x_test = np.array([[4.1]])
# exemple (4.1 heures)

# probabilité
prob = model.predict_proba(x_test)  
# retourne [P(classe 0), P(classe 1)]

# classe prédite
prediction = model.predict(x_test)

print("\nFor 4.1 hours:")
print("Probability of pass:", prob[0][1])  
# probabilité de réussite

print("Prediction:", prediction[0])  
# 0 ou 1