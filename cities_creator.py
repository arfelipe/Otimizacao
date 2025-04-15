import numpy as np
import matplotlib.pyplot as plt

# Definir o número de cidades
num_cities = 100

# Gerar coordenadas aleatórias para as cidades
cities = np.random.rand(num_cities, 2) * 100

# Salvar as coordenadas em um arquivo txt
with open('cities.txt', 'w') as f:
    for city in cities:
        f.write(f"{city[0]}, {city[1]}\n")

print("Arquivo 'cities.txt' criado com sucesso.")

# Plotar a disposição das cidades
plt.figure(figsize=(10, 10))
plt.scatter(cities[:, 0], cities[:, 1], color='blue')
plt.title('Disposição das Cidades')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
