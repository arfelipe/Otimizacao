import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_cities(file_path):
    pontos_x = []
    pontos_y = []
    with open(file_path, 'r') as file:
        for linha in file:
            coordenadas = linha.strip().replace(',', ' ').split()
            if len(coordenadas) == 2:
                try:
                    x = float(coordenadas[0])
                    y = float(coordenadas[1])
                    pontos_x.append(x)
                    pontos_y.append(y)
                except ValueError:
                    continue
    # Plotando os pontos
    plt.scatter(pontos_x, pontos_y, color='black', marker='o')
    plt.xlabel('COORDENADA X')
    plt.ylabel('COORDENADA Y')
    plt.title('CIDADES')
    plt.grid(True)
    plt.xlim(min(pontos_x) - 2, max(pontos_x) + 2)
    plt.ylim(min(pontos_y) - 2, max(pontos_y) + 2)
    plt.show()

class GeneticAlgorithmTSP:
    def __init__(self, num_cities, distance_matrix, pop_size, num_generations, crossover_rate, mutation_rate, elitism_rate):
        self.num_cities = num_cities
        self.distance_matrix = distance_matrix
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.population = self.initialize_population()
        self.best_route = None
        self.best_route_length = float('inf')
        self.fitness_history = []

    def initialize_population(self):
        population = [np.random.permutation(self.num_cities) for _ in range(self.pop_size)]
        return population

    def run(self):
        for generation in range(self.num_generations):
            fitnesses = [self.fitness(ind) for ind in self.population]
            new_population = []
            elite_size = int(self.elitism_rate * self.pop_size)
            elites = sorted(zip(self.population, fitnesses), key=lambda x: x[1])[:elite_size]
            new_population.extend([elite[0] for elite in elites])
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection(fitnesses)
                parent2 = self.tournament_selection(fitnesses)
                if np.random.rand() < self.crossover_rate:
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                if np.random.rand() < self.mutation_rate:
                    child = self.swap_mutation(child)
                new_population.append(child)
            self.population = new_population
            best_fitness_index = np.argmin(fitnesses)
            if fitnesses[best_fitness_index] < self.best_route_length:
                self.best_route_length = fitnesses[best_fitness_index]
                self.best_route = self.population[best_fitness_index]
            self.fitness_history.append(self.best_route_length)
        return self.best_route, self.best_route_length

    def fitness(self, solution):
        route_length = sum(self.distance_matrix[solution[i], solution[i+1]] for i in range(len(solution)-1))
        route_length += self.distance_matrix[solution[-1], solution[0]]
        return route_length

    def tournament_selection(self, fitnesses, k=3):
        selected_indices = np.random.choice(range(len(fitnesses)), k, replace=False)
        selected_indices_sorted = sorted(selected_indices, key=lambda x: fitnesses[x])
        return self.population[selected_indices_sorted[0]]

    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for city in parent2:
            if city not in child:
                while child[pointer] is not None:
                    pointer += 1
                child[pointer] = city
        return child
    def swap_mutation(self, solution):
        i, j = np.random.choice(range(len(solution)), 2, replace=False)
        solution[i], solution[j] = solution[j], solution[i]
        return solution

def run_experiment(pop_size, num_generations, crossover_rate, mutation_rate, elitism_rate, num_runs):
    try:
        cities = pd.read_csv("cities.txt", sep=",", header=None, names=["x", "y"])
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return
    num_cities = cities.shape[0]
    coordinates = cities.values.astype(float)  # Convert to float to avoid TypeError
    # Matriz de distâncias
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :], axis=-1)
    
    all_fitness_histories = []
    best_overall_route = None
    best_overall_length = float('inf')
    
    for run in range(num_runs):
        ga_tsp = GeneticAlgorithmTSP(num_cities, distance_matrix, pop_size, num_generations, crossover_rate, mutation_rate, elitism_rate)
        best_route, best_route_length = ga_tsp.run()
        
        all_fitness_histories.append(ga_tsp.fitness_history)
        
        if best_route_length < best_overall_length:
            best_overall_length = best_route_length
            best_overall_route = best_route
    
    all_fitness_histories = np.array(all_fitness_histories)
    
    mean_fitness = np.mean(all_fitness_histories, axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_generations + 1), mean_fitness, marker="o", linestyle="-", color="black", label="FITNESS")
    plt.title("FITNESS AO LONGO DAS GERAÇÕES")
    plt.xlabel("GERAÇÃO")
    plt.ylabel("MENOS DISTÂNCIA")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    best_coordinates = coordinates[best_overall_route]
    
    plt.figure(figsize=(10, 5))
    plt.plot(best_coordinates[:, 0], best_coordinates[:, 1], marker="o", linestyle="-", color="black")
    plt.scatter(best_coordinates[0, 0], best_coordinates[0, 1], color="red", s=100)
    
    plt.title(f"MELHOR COMPRIMENTO DE ROTA ENCONTRADO: {best_overall_length}")
    plt.xlabel("COORDENADA X")
    plt.ylabel("COORDENADA Y")
    
    plt.grid(False)
    
    plt.show()
    
    # Printar os resultados no terminal
    print(f"Melhor distância encontrada: {best_overall_length}")
    print(f"Desvio padrão das distâncias: {np.std(all_fitness_histories)}")
    print(f"Variância das distâncias: {np.var(all_fitness_histories)}")

if __name__ == "__main__":
    plot_cities('cities.txt')
    
# Cenário de execução do algoritmo genético para o TSP - Cenário 1 (Equilíbrio)
run_experiment(
   pop_size=200,
   num_generations=100,
   crossover_rate=0.9,
   mutation_rate=0.05,
   elitism_rate=0.2,
   num_runs=10
)

# Cenário de execução do algoritmo genético para o TSP - Cenário 2 (Otimização)
run_experiment(
   pop_size=300,  # Aumentar o tamanho da população
   num_generations=200,  # Aumentar o número de gerações
   crossover_rate=0.95,  # Aumentar a taxa de cruzamento
   mutation_rate=0.01,  # Diminuir a taxa de mutação
   elitism_rate=0.3,  # Aumentar a taxa de elitismo
   num_runs=10
)
