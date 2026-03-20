import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt

# Preparando os dados
data = pd.DataFrame({
    'tem_escamas': [1, 1, 0, 0],
    'vive_na_agua':[1, 0, 1, 0],
    'Classes':['peixe', 'reptil', 'anfibio', 'mamifero']
})

X = data[['tem_escamas', 'vive_na_agua']]
y = data['Classes']

# Criando a arvore
arvore = DecisionTreeClassifier(max_depth=3)
arvore.fit(X,y)

# Visualizando o mapa
plt.figure(figsize=(10,6))
plot_tree(arvore, feature_names= X.columns, class_names=y.unique(), filled=True,)
plt.show()
