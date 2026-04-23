
#1.Dataset
import numpy as np
data = np.loadtxt("../Datasets/data.csv", delimiter=",", skiprows=1)

X = data[:, 0]
Y = data[:, 1]

n = len(X)

# Régression Linéaire FROM SCRATCH

n = len(X)  #nbr de donnees

# 2. PARAMÈTRES DU MODÈLE
# a = pente, b = biais
a = 0.0
b = 0.0

# 3. MODÈLE (hypothèse)
def model(x, a, b):
    return a * x + b

# 4. FONCTION DE COÛT (MSE)
def cost_function(X, Y, a, b):
    total_error = 0
    
    for i in range(n):
        y_pred = model(X[i], a, b)
        total_error += (y_pred - Y[i]) ** 2
    
    return total_error / (2 * n)   # 1/(2n)

# 5. GRADIENT DESCENT
def gradient_descent(X, Y, a, b, learning_rate, iterations):
    
    for iteration in range(iterations):
        da = 0
        db = 0
        
        for i in range(n):
            y_pred = model(X[i], a, b)
            
            # dérivées partielles
            da += (y_pred - Y[i]) * X[i]
            db += (y_pred - Y[i])
        
        # moyenne
        da = da / n
        db = db / n
        
        # mise à jour des paramètres
        a = a - learning_rate * da
        b = b - learning_rate * db
        
        # affichage du cout
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: coût = {cost_function(X, Y, a, b)}")
    
    return a, b

# 6. ENTRAÎNEMENT
learning_rate = 0.00001
iterations = 500

a, b = gradient_descent(X, Y, a, b, learning_rate, iterations)

print("\nParamètres appris :")
print("a =", a)
print("b =", b)


# 7. PRÉDICTION
x_test = 85
y_pred = model(x_test, a, b)

print("\nPrediction pour x =", x_test, ":", y_pred)

# 8. COÛT FINAL
final_cost = cost_function(X, Y, a, b)
print("\nCoût final :", final_cost)