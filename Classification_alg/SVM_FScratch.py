import numpy as np
import pandas as pd

# 1. LOAD DATA
data = pd.read_csv("../Datasets/Classification_Data.csv")

X = data["hours"].values
y = data["label"].values

# transformation labels (-1, +1) pour SVM
y = np.where(y == 0, -1, 1)

# 2. INITIALISATION
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 2000

# 3. TRAINING (SVM simple)
for i in range(epochs):

    for idx, x_i in enumerate(X):

        condition = y[idx] * (w * x_i + b)

        # si mal classé
        if condition < 1:
            w = w + learning_rate * (y[idx] * x_i)
            b = b + learning_rate * y[idx]
        else:
            w = w - learning_rate * (0.001 * w)

# 4. PREDICTION
def predict_svm(x):
    return 1 if (w * x + b) >= 0 else 0

# 5. TEST
print("SVM prediction for 4.5:", predict_svm(4.5))
print("SVM prediction for 6:", predict_svm(6))