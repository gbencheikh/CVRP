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
    
class Particle:
        def __init__(self, num_customers, num_vehicles, capacity, distance_matrix, demands):
            self.position = np.random.permutation(num_customers)  # Initialize as permutation
            self.best_position = np.copy(self.position)
            self.velocity = np.zeros(num_customers)  
            self.best_cost = float('inf')
            self.cost = float('inf')
            self.num_vehicles = num_vehicles
            self.capacity = capacity
            self.distance_matrix = distance_matrix
            self.demands = demands

        def decode(self):
            routes = [[] for _ in range(self.num_vehicles)]
            vehicle_index = 0
            load = 0
            for customer in self.position:
                demand = next((d for id_ville, d in self.demands if id_ville == customer), 0)
                if load + demand > self.capacity:
                    vehicle_index += 1
                    load = 0
                    if vehicle_index >= self.num_vehicles:
                        break  # Plus de véhicules disponibles
                routes[vehicle_index].append(customer)
                load += demand

            return routes 
            
        # Methode pour évaluer le cout d'une solution (un ensemble de routes)
        def evaluate_cost(self, NODE_COORD_SECTION):
            routes = self.decode()

            self.cost = evaluer_solution(routes, NODE_COORD_SECTION)
            
            # Mettre à jour le meilleur personnel
            if self.cost < self.best_cost:
                self.best_cost = self.cost
                self.best_position = np.copy(self.position)

class Partical_Swarm_Optimization:
    def __init__(self, file_path):
        # Load the data
        self.NODE_COORD_SECTION, self.demandes = read_file(file_path)
        self.num_customers = len(self.NODE_COORD_SECTION)

    def display_routes(self, position):
        routes = [[] for _ in range(self.num_vehicles)]
        vehicle_index = 0
        load = 0
        
        for customer in position:
            demand = next((d for id_ville, d in self.demands if id_ville == customer), 0)
            if load + demand > self.capacity:
                vehicle_index += 1
                load = 0
                if vehicle_index >= self.num_vehicles:
                    break  # Plus de véhicules disponibles
            routes[vehicle_index].append(customer)
            load += demand
        
        return routes

    def run(self, num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight):
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.num_vehicles = nb_vehicules
        self.capacity = capacite_vehicule
        self.distance_matrix = self.NODE_COORD_SECTION
        self.demands = self.demandes
        self.particles = [Particle(self.num_customers, self.num_vehicles, self.capacity, self.distance_matrix, self.demands) for _ in range(self.num_particles)]
        self.global_best_position = None
        self.global_best_cost = float('inf')

        for _ in range(self.num_iterations):
            for particle in self.particles:
                particle.evaluate_cost(self.NODE_COORD_SECTION)
                if particle.cost < self.global_best_cost:
                    self.global_best_cost = particle.cost
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                # Mettre à jour la vitesse en fonction de la dynamique du PSO
                inertia = inertia_weight * particle.velocity
                cognitive_component = cognitive_weight * random.random() * (particle.best_position - particle.position)
                social_component = social_weight * random.random() * (self.global_best_position - particle.position)
                particle.velocity = inertia + cognitive_component + social_component
                
                # Au lieu de mettre à jour directement les positions, les gérer comme des permutations
                new_position = particle.position + particle.velocity
                new_position = np.clip(new_position, 0, self.num_customers - 1)
                particle.position = np.argsort(new_position)  # Trier comme une permutation
        
        return self.display_routes(self.global_best_position), self.global_best_cost
    
class Ant:
    def __init__(self, graph, pheromone_matrix, alpha, beta, capacity, rho, Q):
        self.graph = graph
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.capacity = capacity
        self.tour = []
        self.rho = rho  # Taux d'évaporation
        self.Q = Q  # Constante de dépôt de phéromones

    def ant_tour(self, remaining_customers):
        self.tour = ['depot']
        current_capacity = 0

        while remaining_customers:
            current_node = self.tour[-1]
            probabilities = self.calculate_probabilities(current_node, remaining_customers)
            next_node = self.select_next_node(probabilities)

            if next_node == 'depot':
                self.tour.append('depot')
                current_capacity = 0
                continue

            # Extraire le numéro du client
            customer_number = int(next_node.split('_')[1])
            demand = self.graph.nodes[next_node]['demand']

            if current_capacity + demand <= self.capacity:
                current_capacity += demand
                self.tour.append(next_node)
                remaining_customers.remove(customer_number)
            else:
                self.tour.append('depot')
                current_capacity = 0

        self.tour.append('depot')

    def calculate_probabilities(self, current_node, remaining_customers):
        pheromone_values = []
        heuristic_values = []
        index_customer = []

        for customer in remaining_customers:
            next_node = f'customer_{customer}'
            if self.graph.has_edge(current_node, next_node):
                pheromone = self.pheromone_matrix.get((current_node, next_node), 1.0)
                distance = self.graph[current_node][next_node]['weight']
                heuristic = 1 / distance if distance > 0 else 0

                index_customer.append(customer)
                pheromone_values.append(pheromone)
                heuristic_values.append(heuristic)

        # Calcul des valeurs totales pour normaliser les probabilités
        total_pheromones = sum([(pheromone ** self.alpha) * (heuristic ** self.beta) for pheromone, heuristic in zip(pheromone_values, heuristic_values)])

        probabilities = []
        for customer, pheromone, heuristic in zip(index_customer, pheromone_values, heuristic_values):
            if total_pheromones > 0:
                probability = (pheromone ** self.alpha) * (heuristic ** self.beta) / total_pheromones
            else:
                probability = 0
            probabilities.append((customer, probability))

        return probabilities

    def select_next_node(self, probabilities):
        if not probabilities:
            return 'depot'
        customers, probs = zip(*probabilities)
        selected_customer = random.choices(customers, weights=probs, k=1)[0]
        return f'customer_{selected_customer}'

    def calculate_tour_cost(self):
        total_cost = 0
        for i in range(len(self.tour) - 1):
            node_i = self.tour[i]
            node_j = self.tour[i + 1]
            if self.graph.has_edge(node_i, node_j):
                distance = self.graph[node_i][node_j]['weight']
                total_cost += distance
            else:
                # Si l'arête n'existe pas, attribuer une distance très élevée
                total_cost += 1e6
        return total_cost
    
class Ant_Colony_Optimization:
    def __init__(self, file_path, ):
        self.num_customers, self.vehicle_capacity, depot, self.customer_data = read_VRP_input_file(file_path)
        self.graph = create_graph(self.num_customers, depot, self.customer_data)
        self.pheromone_matrix = self.initialize_pheromones()

    def initialize_pheromones(self):
        pheromones = {}
        nodes = list(self.graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pheromones[(nodes[i], nodes[j])] = 1.0
                pheromones[(nodes[j], nodes[i])] = 1.0  # Pour arêtes bidirectionnelles
        return pheromones

    def update_pheromones(self, ants):
        # Évaporation des phéromones
        for edge in self.pheromone_matrix:
            self.pheromone_matrix[edge] *= (1 - self.evaporation_rate)

        # Dépôt de nouvelles phéromones
        for ant in ants:
            cost = ant.calculate_tour_cost()
            if cost == 0:
                continue  # Éviter la division par zéro
            pheromone_deposit = self.Q / cost
            for i in range(len(ant.tour) - 1):
                edge = (ant.tour[i], ant.tour[i + 1])
                if edge in self.pheromone_matrix:
                    self.pheromone_matrix[edge] += pheromone_deposit
                # Pour une arête bidirectionnelle
                reverse_edge = (ant.tour[i + 1], ant.tour[i])
                if reverse_edge in self.pheromone_matrix:
                    self.pheromone_matrix[reverse_edge] += pheromone_deposit

    def run(self, num_iterations=100, num_ants=100, alpha=1, beta=3, evaporation_rate=0.2, Q=70):
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q  # Constante de dépôt de phéromones

        best_cost = float('inf')
        best_solution = []

        for _ in range(num_iterations):
            ants = [Ant(self.graph, self.pheromone_matrix, self.alpha, self.beta, self.vehicle_capacity, self.evaporation_rate, self.Q) for _ in range(self.num_ants)]
            for ant in ants:
                remaining_customers = set(cust[0] for cust in self.customer_data if cust[0] != 1)  # Exclure le dépôt
                ant.ant_tour(remaining_customers)

            self.update_pheromones(ants)

            for ant in ants:
                cost = ant.calculate_tour_cost()
                if cost < best_cost:
                    best_cost = cost
                    best_solution = ant.tour

        return best_solution, best_cost