import numpy as np  # calculs numériques
import pandas as pd  # lecture des données

# 1. LOAD DATA
data = pd.read_csv("../Datasets/polynomial_data.csv")  # lecture fichier

x = data["x"].values  # variable d'entrée
y = data["y"].values  # variable cible

n = len(x)  # nombre d'exemples

# 2. INITIALISATION DES PARAMÈTRES
w0 = 0.0  # biais
w1 = 0.0  # coefficient x
w2 = 0.0  # coefficient x^2


# 3. HYPERPARAMÈTRES
learning_rate = 0.0001  # petit pas car x^2 peut exploser
epochs = 5000  # itérations


# 4. DESCENTE DE GRADIENT
for i in range(epochs):

    # PRÉDICTION DU MODÈLE
    y_pred = w0 + w1*x + w2*(x**2)

    # FONCTION DE COÛT (MSE)
    cost = (1/n) * np.sum((y - y_pred) ** 2)

    # GRADIENTS
    dw0 = (-2/n) * np.sum(y - y_pred)
    dw1 = (-2/n) * np.sum(x * (y - y_pred))
    dw2 = (-2/n) * np.sum((x**2) * (y - y_pred))

    # MISE À JOUR
    w0 = w0 - learning_rate * dw0
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2

    # AFFICHAGE PROGRESSION
    if i % 500 == 0:
        print(f"Epoch {i}, Cost: {cost:.4f}")


# 5. RÉSULTATS
print("\nFinal model:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)


# 6. TEST
x_test = 12
prediction = w0 + w1*x_test + w2*(x_test**2)

print("\nPrediction for x=12:", prediction)