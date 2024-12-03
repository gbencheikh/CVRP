from utils import *
import time
from params import * 

class Hill_Climbing:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.nb_villes = len(self.NODE_COORD_SECTION)
        # Generate initial solution
        self.meilleure_solution = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)

    def run(self):
        self.meilleure_distance = evaluer_solution(self.meilleure_solution, self.NODE_COORD_SECTION)

        while True:
            voisins = generer_voisins(self.meilleure_solution)
            meilleur_voisin = None
            meilleure_distance_voisin = float('inf')

            for voisin in voisins:
                distance_voisin = evaluer_solution(voisin, self.NODE_COORD_SECTION)

                if distance_voisin < meilleure_distance_voisin:
                    meilleur_voisin = voisin
                    meilleure_distance_voisin = distance_voisin

            if meilleure_distance_voisin < self.meilleure_distance:
                self.meilleure_solution = meilleur_voisin
                self.meilleure_distance = meilleure_distance_voisin
            else:
                break  

        return self.meilleure_solution, self.meilleure_distance

class Tabu_Search:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.nb_villes = len(self.NODE_COORD_SECTION)
        # Generate initial solution
        self.best_solution = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)
    
    def run(self, max_iterations=100, tabu_tenure=10):
        # Check if initialization was successful
        if self.best_solution is None:
            print("Error: Initial solution is invalid.")
            return None, None
        
        self.best_cost = evaluer_solution(self.best_solution, self.NODE_COORD_SECTION)
        current_solution = self.best_solution
        tabu_list = []
        
        for iteration in range(max_iterations):
            neighborhood = generer_voisins(current_solution)
            best_neighbor = None
            best_neighbor_cost = float('inf')

            # Evaluate neighbors and choose the best one
            for neighbor in neighborhood:
                neighbor_cost = evaluer_solution(neighbor, self.NODE_COORD_SECTION)
                if neighbor_cost < best_neighbor_cost and neighbor not in tabu_list:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

            # Update if the best neighbor is better
            if best_neighbor_cost < self.best_cost:
                self.best_solution = best_neighbor
                self.best_cost = best_neighbor_cost
            
            current_solution = best_neighbor
            tabu_list.append(best_neighbor)

            # Maintain tabu list size
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
                    
        return self.best_solution, self.best_cost

class Simulated_Annealing:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.nb_villes = len(self.NODE_COORD_SECTION)
        # Generate initial solution
        self.best_solution = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)

    def run(self, initial_temperature, cooling_rate, minimal_temperature, max_iter):
        if self.best_solution is None:
            raise ValueError("Failed to initialize a valid solution.")
        
        # Calculate the distance of the initial solution
        self.best_distance = evaluer_solution(self.best_solution, self.NODE_COORD_SECTION)
        current_solution = self.best_solution
        current_distance = self.best_distance
        
        # Define initial temperature and cooling parameters
        temperature = initial_temperature
        
        # Step 4: Simulated Annealing loop
        start_time = time.process_time()  # Start time tracking
        while temperature > minimal_temperature:
            for _ in range(max_iter):
                # Generate a neighbor solution by swapping two random nodes in random routes
                new_solution = generate_neighbor(current_solution, self.demandes, capacite_vehicule)
                new_distance = evaluer_solution(new_solution, self.NODE_COORD_SECTION)
                
                # Calculate the change in distance
                delta_distance = new_distance - current_distance
                
                # Acceptance criteria: if new solution is better or by probability
                if delta_distance < 0 or np.exp(-delta_distance / temperature) > random.random():
                    current_solution = new_solution
                    current_distance = new_distance
                    
                    # Update the best solution if the new solution is better
                    if current_distance < self.best_distance:
                        self.best_solution = current_solution
                        self.best_distance = current_distance
            
            # Cool down the temperature
            temperature *= cooling_rate
         
        return self.best_solution, self.best_distance
    
class Golden_Ball_Algorithm:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.nb_villes = len(self.NODE_COORD_SECTION)
        self.num_customers = len(self.NODE_COORD_SECTION)

    def calculate_distance_matrix(self):
        n = len(self.NODE_COORD_SECTION)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance_matrix[i][j] = distance_entre_villes(self.NODE_COORD_SECTION[i], self.NODE_COORD_SECTION[j])
        return distance_matrix
    
    def initialize_population(self, population_size):
        population = []
        for _ in range(population_size):
            routes = []
            remaining_customers = list(range(1, self.num_customers))  
            while remaining_customers:
                route = []
                load = 0
                # Construire une route en respectant les contraintes de capacité
                for customer in remaining_customers[:]:  # Itérer sur une copie pour permettre des suppressions
                    demand = next(demande for ville, demande in self.demandes if ville == customer)
                    
                    if load + demand <= capacite_vehicule:
                        route.append(customer)
                        load += demand
                        remaining_customers.remove(customer)  # Retirer le client traité de la liste
                
                if route:  # Ajouter la route construite si elle contient des clients
                    routes.append(route)
        
            population.append(routes)
        return population
    
    # Team Division: Divide Population into Teams
    def divide_into_teams(self, population, num_teams):
        random.shuffle(population)
        return [population[i::num_teams] for i in range(num_teams)]

    # Local Search (2-opt) to Improve Solution
    def two_opt(self, route):
        best_route = route
        best_distance = evaluer_solution([best_route], self.NODE_COORD_SECTION)
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                new_distance = evaluer_solution([new_route], self.NODE_COORD_SECTION)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
        return best_route
    
    def run(self, population_size, num_teams, num_generations):
        # Initialize population of players (solutions)
        population = self.initialize_population(population_size)
        
        # Divide the population into teams
        teams = self.divide_into_teams(population, num_teams)
        
        self.best_solution = None
        self.best_distance = float('inf')
        
        # Repeat until termination criterion is met
        for generation in range(num_generations):
            # Execute a season (improve solutions)
            for team in teams:
                # Perform local search or other operations on each team's solutions
                for i in range(len(team)):
                    for route_idx in range(len(team[i])):
                        team[i][route_idx] = self.two_opt(team[i][route_idx])
            
            # Evaluate all players and select the best solution in each team
            for team in teams:
                for individual in team:
                    total_distance = evaluer_solution(individual, self.NODE_COORD_SECTION)
                    if total_distance < self.best_distance:
                        self.best_solution = individual
                        self.best_distance = total_distance
        
        return self.best_solution, self.best_distance
    
class Genetic_Algorithm:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.nb_villes = len(self.NODE_COORD_SECTION)
        # Generate initial solution
        self.meilleure_solution = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)

    # Générer une population initiale
    def generer_population(self, taille_population):
        population = []
        for _ in range(taille_population):
            ok = False
            while ok == False:
                sol = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)
                if verifier_capacite(sol, self.demandes):
                    population.append(sol)
                    ok = True
                else:
                    print("Solution initiale invalide. Génération d'une nouvelle solution.")

        return population
    
    # Sélection par tournoi
    def selection(self, population, scores, k=3):
        selection_ix = random.randint(0, len(population) - 1)
        for ix in random.sample(range(len(population)), k - 1):
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return population[selection_ix]
    
    # Crossover: Crossover d'ordre (OX)
    def crossover(self, parent1, parent2):
        # Parent solutions are lists of routes
        p1 = [ville for route in parent1 for ville in route]
        p2 = [ville for route in parent2 for ville in route]

        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child_p = [None] * size
        child_p[a:b] = p1[a:b]

        ptr = b
        for city in p2[b:] + p2[:b]:
            if city not in child_p:
                if ptr >= size:
                    ptr = 0
                child_p[ptr] = city
                ptr += 1

        # Reconstruct the child solution into routes
        child = [[] for _ in range(nb_vehicules)]
        capacites = [0] * nb_vehicules
        for ville in child_p:
            demande = next((d for id_ville, d in self.demandes if id_ville == ville), None)
            if demande is None:
                continue

            # Try to assign the city to the same vehicle as in parent1 if possible
            assigned = False
            for vehicule in range(nb_vehicules):
                if capacites[vehicule] + demande <= capacite_vehicule:
                    child[vehicule].append(ville)
                    capacites[vehicule] += demande
                    assigned = True
                    break

            if not assigned:
                # If no valid vehicle, distribute to the first that fits
                for vehicule in range(nb_vehicules):
                    if capacites[vehicule] + demande <= capacite_vehicule:
                        child[vehicule].append(ville)
                        capacites[vehicule] += demande
                        break

        return child

    # Mutation: Mutation par échange
    def mutation(self, solution, mutation_rate=0.1):
        for i in range(len(solution)):
            if random.random() < mutation_rate:
                j = random.randint(0, len(solution) - 1)
                solution[i], solution[j] = solution[j], solution[i]
        if not verifier_capacite(solution, self.demandes):
            print("Capacité violée après mutation. Réparation de la solution.")
            # Correction de la violation de capacité sans régénérer la solution entière
            solution = generate_solution(self.nb_villes, nb_vehicules, capacite_vehicule, self.demandes)
        return solution
    
    def run(self, taille_population, generations, mutation_rate):
        population = self.generer_population(taille_population)
        self.best_solution = None
        self.best_score = float('inf')

        for _ in range(generations):
            scores = [evaluer_solution(individual, self.NODE_COORD_SECTION) for individual in population]

            for i in range(taille_population):
                if scores[i] < self.best_score:
                    self.best_solution = population[i]
                    self.best_score = scores[i]

            # Sélectionner les parents
            new_population = []
            for _ in range(taille_population // 2):
                parent1 = self.selection(population, scores)
                parent2 = self.selection(population, scores)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.append(self.mutation(child1, mutation_rate))
                new_population.append(self.mutation(child2, mutation_rate))
            
            population = new_population

        return self.best_solution, self.best_score