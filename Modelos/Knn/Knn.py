from sklearn.neighbors import NearestNeighbors
import numpy as np

# Pontos cartesianos (x,y)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# Aqui ele vai encontrar os visinhos proximos aos pontos da lista
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

distancia, indices = nbrs.kneighbors(X)

print(distancia)

print(indices)
