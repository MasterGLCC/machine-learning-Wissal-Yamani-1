import numpy as np  # calculs numériques
import pandas as pd  # lecture des données


# 1. LOAD DATA
data = pd.read_csv("../Datasets/Classification_Data.csv")
# lecture du fichier CSV

X = data[["hours"]].values  # variable d'entrée (2D)
y = data["label"].values     # variable cible (0 ou 1)

n_samples = X.shape[0]  # nombre d'exemples


# 2. INITIALISATION
w = 0.0  # poids
b = 0.0  # biais


# 3. HYPERPARAMÈTRES
learning_rate = 0.1  # vitesse d’apprentissage
epochs = 2000        # nombre d’itérations


# 4. FONCTION SIGMOÏDE
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # transforme en probabilité [0,1]


# 5. DESCENTE DE GRADIENT
for i in range(epochs):

    # CALCUL DE z
    z = w * X.flatten() + b  
    # X.flatten() transforme [[x]] en [x]

    # PRÉDICTION (probabilité)
    y_pred = sigmoid(z)


    # FONCTION DE COÛT (log loss)
    cost = (-1/n_samples) * np.sum(
        y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9)
    )
    # 1e-9 évite log(0)


    # GRADIENTS
    dw = (1/n_samples) * np.sum((y_pred - y) * X.flatten())
    db = (1/n_samples) * np.sum(y_pred - y)


    # MISE À JOUR
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # affichage
    if i % 200 == 0:
        print(f"Epoch {i}, Cost: {cost:.4f}")


# 6. RÉSULTATS
print("\nFinal parameters:")
print("w =", w)
print("b =", b)

# 7. TEST
x_test = 4.1 # seuil intéressant

z = w * x_test + b
prob = sigmoid(z)

# décision
prediction = 1 if prob >= 0.5 else 0

print("\nFor 4.1 hours:")
print("Probability:", prob)
print("Prediction:", prediction)