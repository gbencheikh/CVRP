from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt

def read_VRP_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_customers = int(lines[0].split(":")[1].strip())
    vehicle_capacity = int(lines[1].split(":")[1].strip())
    depot_coordinates = tuple(map(int, lines[2].split(":")[1].strip().split(',')))
    
    # Skip the header line and read customer data
    customer_data = [list(map(int, line.strip().split(','))) for line in lines[num_customers:]]

    return num_customers, vehicle_capacity, depot_coordinates, customer_data

class Graph:
    def __init__(self):
        """ default constructor: creates graphs adjancency matrix """
        self.adjacency_list = {}

    def add_edge(self, vertex1, vertex2, weight):
        """ function adds new adge into the graph  """
        edge = (vertex2, weight)
        if vertex1 in self.adjacency_list:
            self.adjacency_list[vertex1].append(edge)
        else:
            self.adjacency_list[vertex1] = [edge]

        if vertex2 not in self.adjacency_list:
            self.adjacency_list[vertex2] = []

    def display(self):
        """ function that displays adjencency matrix of graph  """
        for vertex, edges in self.adjacency_list.items():
            edge_str = ", ".join([f"{neighbor}({weight})" for neighbor, weight in edges])
            print(f"{vertex}: {edge_str}")

    def draw_graph(self):
        """ display graph """
        G = nx.DiGraph()
        for vertex, edges in self.adjacency_list.items():
            for neighbor, weight in edges:
                G.add_edge(vertex, neighbor, weight=weight)

        pos = nx.spring_layout(G)  # Layout to orginire nodes
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        labels = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color="skyblue")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()

import random 
import numpy as np 
import copy 

# Function to load data from file
def read_file(file_path):
    with open(file_path, 'r') as f:
        lignes = f.readlines()

    NODE_COORD_SECTION = []
    demandes = []

    reading_coords = False
    reading_demand = False

    for ligne in lignes:
        ligne = ligne.strip()
        
        if ligne == "NODE_COORD_SECTION":
            reading_coords = True
            continue
        elif ligne == "DEMAND_SECTION":
            reading_coords = False
            reading_demand = True
            continue
        elif ligne == "DEPOT_SECTION":
            break  # End of sections to read

        if reading_coords:
            elements = ligne.split()
            if len(elements) >= 3:
                id_ville = int(elements[0])
                coord_x = float(elements[1])
                coord_y = float(elements[2])
                NODE_COORD_SECTION.append((id_ville, coord_x, coord_y))

        if reading_demand:
            elements = ligne.split()
            if len(elements) >= 2:
                id_ville = int(elements[0])
                demande = int(elements[1])
                demandes.append((id_ville, demande))

    return NODE_COORD_SECTION, demandes

def generate_solution(nb_villes, nb_vehicules, capacite_vehicule, demandes):
    solution = [[] for _ in range(nb_vehicules)]
    villes = list(range(1, nb_villes + 1))  # Start from 1 to skip depot (node 0)
    random.shuffle(villes)
    
    capacites = [0] * nb_vehicules

    for ville in villes:
        demande = next((demande for id_ville, demande in demandes if id_ville == ville), None)
        
        if demande is None:
            continue
        assigned = False
        for vehicule in range(nb_vehicules):
            if capacites[vehicule] + demande <= capacite_vehicule:
                solution[vehicule].append(ville)
                capacites[vehicule] += demande
                assigned = True
                break
        
        if not assigned:
            continue

    return solution

def distance_entre_villes(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def distance_totale(solution, NODE_COORD_SECTION):
    total_distance = 0
    for vehicule in solution:
        if len(vehicule) == 0:
            continue
        depot_coord = NODE_COORD_SECTION[0][1:3]  # Starting point (depot)
        # Distance from depot to the first city and back to depot
        total_distance += distance_entre_villes(depot_coord, NODE_COORD_SECTION[vehicule[0] - 1][1:3])
        for i in range(len(vehicule) - 1):
            ville1 = vehicule[i]
            ville2 = vehicule[i + 1]
            coord1 = NODE_COORD_SECTION[ville1 - 1][1:3]
            coord2 = NODE_COORD_SECTION[ville2 - 1][1:3]
            total_distance += distance_entre_villes(coord1, coord2)
        total_distance += distance_entre_villes(NODE_COORD_SECTION[vehicule[-1] - 1][1:3], depot_coord)
    return total_distance

def generer_voisins(solution):
    voisins = []
    for vehicule in range(len(solution)):
        if len(solution[vehicule]) < 2:
            continue
        for i in range(len(solution[vehicule]) - 1):
            for j in range(i + 1, len(solution[vehicule])):
                voisin = copy.deepcopy(solution)
                voisin[vehicule][i], voisin[vehicule][j] = voisin[vehicule][j], voisin[vehicule][i]
                voisins.append(voisin)
    return voisins

def evaluer_solution(solution, NODE_COORD_SECTION):
    return distance_totale(solution, NODE_COORD_SECTION)

# Displaying the routes and total cost
def afficher_solution(solution, distance):
    for i, vehicule in enumerate(solution, 1):
        if vehicule:  # If the vehicle has assigned cities
            villes = " ".join(map(str, vehicule))
            print(f"Route #{i}: {villes}")
    print(f"Le coût de la meilleure solution: {distance:.4f}")