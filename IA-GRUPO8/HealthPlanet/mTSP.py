import itertools
import random

# Function to generate initial population for MTSP
import random
from Graph import Grafo
import datetime


pesos = {1953447602:(1,1),2062531977:(1,3),2524924343:(4,4),4800834784:(12,2),2183399416:(1,9)} # peso e limite de tempo
g = Grafo.load_graph("Braga")

inicio = "2590130489"
def generate_population(pop_size, n, m, cities_list):
    population = []

    for _ in range(pop_size):
        cities_copy = cities_list[:]
        random.shuffle(cities_copy)  # Shuffle the cities randomly

        chrom = []
        groups = [[] for _ in range(m)]  # Create 'm' empty groups

        for city in cities_copy:
            # Randomly assign cities to groups
            idx = random.randint(0, m - 1)
            groups[idx].append(city)

        # Add cities for each group into the chromosome
        for i, group in enumerate(groups):
            chrom.extend(group)
            chrom.append(-(m - (i)))  # Delimiter for salesman ID

        population.append(chrom)

    return population

# Function to calculate fitness for MTSP (update based on your criteria)
def calculate_fitness_show(chromosome, n, m):
    salesman_cities = [[] for _ in range(m)]
    current_salesman = 0

    for gene in chromosome:
        if gene < 0:
            current_salesman = abs(gene) - 1
        else:
            salesman_cities[current_salesman].append(gene)

    fit = 0
    print(salesman_cities)
    for conunto in salesman_cities: # conunto = entregas do estafeta
        pesoTotal = 0
        maxTempo = float('inf')
        current_best_fit = float('inf')
        encs = []
        for e in conunto: # entrega no conjunto
            pesoTotal = pesoTotal + pesos[e][0]
            maxTempo = min(maxTempo, pesos[e][1])
            encs.append((str(e),pesos[e][0]))

        l = ["carro","mota","bicicleta"]
        print(encs)

        for v in l:
            caminho,custo,custoT,custoP = g.fazer_entrega(inicio,encs,'cunif',v)
            #print("Tempo a não ultrapassar: " + str(maxTempo))
            #print("Peso: " + str(pesoTotal))
            #print(encs)
            #print(caminho)
            #print(str(custo) + " " + str(custoT) + " " + str(custoP) +" " + v)
            if (custoT > maxTempo):
                current_best_fit = min(10000000,current_best_fit)
            elif v=="carro" and pesoTotal > 100:
                current_best_fit = min(10000000,current_best_fit)
            elif v=="mota" and pesoTotal > 20:
                current_best_fit = min(10000000,current_best_fit)
            elif v=="bicicleta" and pesoTotal > 5:
                current_best_fit = min(10000000,current_best_fit)
            else:
                new_fit = min(custoP * 0.85 + custoT * 0.15, current_best_fit)
                if new_fit < current_best_fit:  # Only update when a better fit is found
                    current_best_fit = new_fit
                    selected_vehicle = v  # Update the selected vehicle

        fit = fit + current_best_fit
        print("Veiculo:",selected_vehicle)

    print(fit)
    return fit
def calculate_fitness(chromosome, n, m):
    salesman_cities = [[] for _ in range(m)]
    current_salesman = 0

    for gene in chromosome:
        if gene < 0:
            current_salesman = abs(gene) - 1
        else:
            salesman_cities[current_salesman].append(gene)

    fit = 0

    for conunto in salesman_cities: # conunto = entregas do estafeta
        pesoTotal = 0
        maxTempo = float('inf')
        current_best_fit = float('inf')
        encs = []
        for e in conunto: # entrega no conjunto
            pesoTotal = pesoTotal + pesos[e][0]
            maxTempo = min(maxTempo, pesos[e][1])
            encs.append((str(e),pesos[e][0]))

        l = ["carro","mota","bicicleta"]
        if len(encs) > 0:
            for v in l:
                caminho,custo,custoT,custoP = g.fazer_entrega(inicio,encs,'cunif',v)
                #print("Tempo a não ultrapassar: " + str(maxTempo))
                #print("Peso: " + str(pesoTotal))
                #print(encs)
                #print(caminho)
                #print(str(custo) + " " + str(custoT) + " " + str(custoP) +" " + v)
                if (custoT > maxTempo):
                    current_best_fit = min(10000000,current_best_fit)
                elif v=="carro" and pesoTotal > 100:
                    current_best_fit = min(10000000,current_best_fit)
                elif v=="mota" and pesoTotal > 20:
                    current_best_fit = min(10000000,current_best_fit)
                elif v=="bicicleta" and pesoTotal > 5:
                    current_best_fit = min(10000000,current_best_fit)
                else:
                    current_best_fit = min (custoP*0.85+custoT*0.15,current_best_fit)

            fit = fit + current_best_fit

    print(fit)
    return fit

# Tournament Selection for MTSP
def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament = random.choices(population, k=tournament_size)
        tournament_fitness = [fitness_scores[population.index(chrom)] for chrom in tournament]
        selected.append(tournament[tournament_fitness.index(min(tournament_fitness))])
    return selected

# Crossover respecting salesman separators for MTSP
def find_delimiter_positions(chromosome):
    delimiter_positions = []
    for i, gene in enumerate(chromosome):
        if gene < 0:
            delimiter_positions.append(i)
    return delimiter_positions

# Crossover respecting salesman separators for MTSP (without vehicles)
def crossover(parent1, parent2):
    delimiter_positions_parent1 = find_delimiter_positions(parent1)
    delimiter_positions_parent2 = find_delimiter_positions(parent2)

    crossover_points_parent1 = sorted(random.sample(delimiter_positions_parent1, 2))
    crossover_points_parent2 = sorted(random.sample(delimiter_positions_parent2, 2))

    child1 = parent1[:]
    child2 = parent2[:]

    # Swap genes between crossover points
    child1[crossover_points_parent1[0] : crossover_points_parent1[1]] = parent2[
        crossover_points_parent2[0] : crossover_points_parent2[1]
    ]
    child2[crossover_points_parent2[0] : crossover_points_parent2[1]] = parent1[
        crossover_points_parent1[0] : crossover_points_parent1[1]
    ]

    return child1, child2

# Mutation respecting salesman separators for MTSP (without vehicles)
def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome[:]
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            if mutated_chromosome[i] < 0:
                continue  # Do not mutate separators
            swap_index = random.randint(0, len(mutated_chromosome) - 1)
            while mutated_chromosome[swap_index] < 0:
                swap_index = random.randint(0, len(mutated_chromosome) - 1)
            mutated_chromosome[i], mutated_chromosome[swap_index] = (
                mutated_chromosome[swap_index],
                mutated_chromosome[i],
            )
    return mutated_chromosome
# Update the genetic algorithm function to handle MTSP with correct selection, crossover, mutation
def genetic_algorithm_mtsp(population_size, n, m, generations, mutation_rate, tournament_size, cities_list):
    chrom_length = n + m - 1
    population = generate_population(population_size, n, m, cities_list)
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitness_scores = [calculate_fitness(chrom, n, m) for chrom in population]

        # Track the best solution
        max_fitness = min(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitness_scores.index(max_fitness)]

        # Rest of your genetic algorithm logic remains unchanged

    return best_solution, population
# Example usage:
population_size = 10
num_cities = 5
num_salesmen = 2
num_generations = 3
mutation_rate = 0.1
tournament_size = 3

# Assuming 'cities' is a list containing all 'num_cities'
cities =[1953447602,2062531977,2524924343,4800834784,2183399416]
start_time = datetime.datetime.now()  # Get current date and time before the operation
best_solution, final_population = genetic_algorithm_mtsp(
    population_size, num_cities, num_salesmen, num_generations, mutation_rate, tournament_size, cities
)
end_time = datetime.datetime.now()
print("Best Solution:", best_solution)
print("Final Population:", final_population)
execution_time = end_time - start_time  # Calculate the time difference
print(f"Execution Time: {execution_time}")

if calculate_fitness_show(best_solution,num_cities,num_salesmen) > 10000000:
    print("Não obtemos uma solução possível")

