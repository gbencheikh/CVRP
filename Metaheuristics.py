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
