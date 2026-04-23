import numpy as np
import pandas as pd

# 1. LOAD DATA
data = pd.read_csv("../Datasets/Classification_Data.csv")

X = data["hours"].values
y = data["label"].values

# 2. FUNCTION DISTANCE
def distance(a, b):
    return abs(a - b)  # distance 1D simple

# 3. KNN PREDICTION
def predict_knn(x_new, k=3):

    distances = []

    # calcul distance avec tous les points
    for i in range(len(X)):
        d = distance(x_new, X[i])
        distances.append((d, y[i]))

    # tri par distance
    distances.sort(key=lambda x: x[0])

    # prendre les k voisins
    neighbors = distances[:k]

    # vote majoritaire
    votes = [label for _, label in neighbors]

    return 1 if sum(votes) > k/2 else 0


# 4. TEST
print("KNN prediction for 4.5:", predict_knn(4.5))
print("KNN prediction for 6:", predict_knn(6))