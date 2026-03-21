from sklearn import tree

# Separando açai do cupuaçu em que [peso, cor: 1(roxo), 2(branco)]

caracteristica = [[15, 1], [30, 1], [200, 2],[350, 2]]
resposta = ['Açai', 'Açai', 'Cupuaçu', 'Cupaçu']

# Criando a arvore
arvore = tree.DecisionTreeClassifier()

# Treinando a arvore
arvore.fit(caracteristica, resposta)

# Colocando caracteristicas da fruta desconhecida
nova_fruta = [[234, 2]]

# Com base no treinamento, decide qual fruta é
previsão = arvore.predict(nova_fruta)

# Mostrando o resultado
print(f'Essa fruta é {previsão[0]}')
