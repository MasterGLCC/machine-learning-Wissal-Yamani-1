import numpy as np  # bibliothèque pour les calculs numériques
import random  # bibliothèque pour les choix aléatoires


# environnement simple sous forme de grille
# 0 = état normal
# 1 = objectif final

grid = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
]


# nombre de lignes
n_rows = len(grid)

# nombre de colonnes
n_cols = len(grid[0])


# actions possibles
# 0 = haut
# 1 = bas
# 2 = gauche
# 3 = droite

actions = [0, 1, 2, 3]


# création de la Q-table
# dimensions :
# lignes * colonnes * nombre actions

Q = np.zeros((n_rows, n_cols, len(actions)))


# paramètres Q-learning

alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # probabilité d'exploration
episodes = 1000  # nombre d'épisodes


# fonction déplacement agent
def move(state, action):

    # récupérer position actuelle
    row, col = state

    # action haut
    if action == 0:
        row = max(row - 1, 0)

    # action bas
    elif action == 1:
        row = min(row + 1, n_rows - 1)

    # action gauche
    elif action == 2:
        col = max(col - 1, 0)

    # action droite
    elif action == 3:
        col = min(col + 1, n_cols - 1)

    # retourner nouvel état
    return (row, col)


# fonction récompense
def reward(state):

    # si objectif atteint
    if grid[state[0]][state[1]] == 1:
        return 10

    # sinon petite pénalité
    return -1


# boucle principale apprentissage
for episode in range(episodes):

    # état initial
    state = (0, 0)

    # continuer jusqu'à objectif
    while True:

        # exploration ou exploitation
        if random.uniform(0, 1) < epsilon:

            # choisir action aléatoire
            action = random.choice(actions)

        else:

            # choisir meilleure action
            action = np.argmax(Q[state[0], state[1]])

        # exécuter action
        next_state = move(state, action)

        # recevoir récompense
        r = reward(next_state)

        # récupérer ancienne valeur Q
        old_value = Q[state[0], state[1], action]

        # meilleure valeur future
        future_max = np.max(Q[next_state[0], next_state[1]])

        # mise à jour Q-learning
        Q[state[0], state[1], action] = old_value + alpha * (
            r + gamma * future_max - old_value
        )

        # passer au nouvel état
        state = next_state

        # arrêter si objectif atteint
        if grid[state[0]][state[1]] == 1:
            break


# affichage Q-table finale
print("Q-table finale :\n")

print(Q)


# test de l'agent appris

print("\nchemin appris :")

# état départ
state = (0, 0)

# afficher départ
print(state)

# suivre politique optimale
while grid[state[0]][state[1]] != 1:

    # meilleure action
    action = np.argmax(Q[state[0], state[1]])

    # nouvel état
    state = move(state, action)

    # afficher état
    print(state)