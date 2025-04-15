import numpy as np
import matplotlib.pyplot as plt
import random

# Função Ackley
def ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum1 = x**2 + y**2
    sum2 = np.cos(c * x) + np.cos(c * y)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / 2))
    term2 = -np.exp(sum2 / 2)
    return term1 + term2 + a + np.exp(1)

# Função de avaliação (fitness)
def fitness(individuo):
    return -ackley(individuo[0], individuo[1])  # Minimizar a função Ackley

# Função para criar uma população inicial
def criar_populacao(tam_pop, num_vars):
    return [np.random.uniform(-35, 35, num_vars) for _ in range(tam_pop)]

# Função de seleção por torneio
def selecao_torneio(populacao, fitness_values, k=3):
    selecionados = random.sample(list(zip(populacao, fitness_values)), k)
    selecionados.sort(key=lambda x: x[1], reverse=True)
    return selecionados[0][0]

# Função de cruzamento
def cruzamento(pai1, pai2):
    ponto_corte = random.randint(1, len(pai1)-1)
    filho1 = np.concatenate((pai1[:ponto_corte], pai2[ponto_corte:]))
    filho2 = np.concatenate((pai2[:ponto_corte], pai1[ponto_corte:]))
    return filho1, filho2

# Função de mutação
def mutacao(individuo, taxa_mutacao=0.02):
    if random.random() < taxa_mutacao:
        pos = random.randint(0, len(individuo)-1)
        individuo[pos] = random.uniform(-35, 35)
    return individuo

# Algoritmo Genético
def algoritmo_genetico(tam_pop=30, num_geracoes=40, taxa_cruzamento=0.75, taxa_mutacao=0.02, taxa_elitismo=0.70):
    num_vars = 2  # Número de variáveis (x e y)
    populacao = criar_populacao(tam_pop, num_vars)
    
    # Para armazenar os melhores fitness ao longo das gerações
    melhores_fitness = []
    
    for geracao in range(num_geracoes):
        fitness_values = [fitness(ind) for ind in populacao]
        nova_populacao = []
        elite_size = int(tam_pop * taxa_elitismo)
        elite = sorted(populacao, key=lambda ind: fitness(ind), reverse=True)[:elite_size]
        nova_populacao.extend(elite)
        while len(nova_populacao) < tam_pop:
            pai1 = selecao_torneio(populacao, fitness_values)
            pai2 = selecao_torneio(populacao, fitness_values)
            if random.random() < taxa_cruzamento:
                filho1, filho2 = cruzamento(pai1, pai2)
            else:
                filho1, filho2 = pai1, pai2
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))
        populacao = nova_populacao[:tam_pop]
        
        # Armazenar o melhor fitness da geração atual
        melhores_fitness.append(max(fitness_values))
    
    fitness_values = [fitness(ind) for ind in populacao]
    melhor_individuo = populacao[np.argmax(fitness_values)]
    
    return melhor_individuo, max(fitness_values), melhores_fitness

# Exemplo de uso
melhor_solucao, melhor_fitness, melhores_fitness_geracoes = algoritmo_genetico()
print("Melhor solução:", melhor_solucao)
print("Melhor fitness:", melhor_fitness)

# Geração dos dados para o gráfico da função Ackley
x = np.linspace(-35, 35, 400)
y = np.linspace(-35, 35, 400)
X, Y = np.meshgrid(x, y)
Z = ackley(X, Y)

# Criação do gráfico de superfície e contorno da função Ackley
fig = plt.figure(figsize=(12, 6))

# Gráfico de superfície 3D
ax = fig.add_subplot(121, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Função Ackley - Gráfico de Superfície')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Gráfico de contorno
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, cmap='viridis')
ax2.set_title('Função Ackley - Gráfico de Contorno')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
fig.colorbar(contour, ax=ax2)

# Exibir os gráficos
plt.show()

# Gráfico do melhor fitness ao longo das gerações
plt.figure(figsize=(10, 5))
plt.plot(melhores_fitness_geracoes)
plt.title('Melhor Fitness ao Longo das Gerações')
plt.xlabel('Geração')
plt.ylabel('Fitness')
plt.grid(True)
plt.show()
