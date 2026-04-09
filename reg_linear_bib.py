
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data.csv")
X = data[["x"]]
Y = data["y"]

# Creation du modele
model = LinearRegression()

# Entrainement
model.fit(X, Y)

a = model.coef_[0]   # pente
b = model.intercept_  # biais

print("\nParamètres appris :")
print("a =", a)
print("b =", b)

# Prediction
x_test = pd.DataFrame({"x": [85]})
y_pred = model.predict(x_test)

print("\nPrediction pour x =", x_test["x"][0], ":", y_pred[0])

# 7. Cout(MSE)
Y_pred_all = model.predict(X)
cost = mean_squared_error(Y, Y_pred_all)

print("\nCoût final (MSE) :", cost)