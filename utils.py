from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional
import networkx as nx
import matplotlib.pyplot as plt
from params import *

def read_VR_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_customers = int(lines[0].split(":")[1].strip())
    vehicle_capacity = int(lines[1].split(":")[1].strip())
    depot_coordinates = tuple(map(int, lines[2].split(":")[1].strip().split(',')))
    
    # Skip the header line and read customer data
    customer_data = [list(map(int, line.strip().split(','))) for line in lines[num_customers:]]

    return num_customers, vehicle_capacity, depot_coordinates, customer_data

def read_VRP_input_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Variables pour stocker les données
    NODE_COORD_SECTION = []
    DEMAND_SECTION = []
    depot = None
    vehicle_capacity = None
    num_customers = 0

    # Flags pour identifier les sections
    reading_coords = False
    reading_demand = False

    for line in lines:
        line = line.strip()

        # Ignorer les lignes vides et les commentaires
        if not line or line.startswith("NAME") or line.startswith("COMMENT") or line.startswith("TYPE") or line.startswith("DIMENSION") or line.startswith("EDGE_WEIGHT_TYPE"):
            continue

        # Lire CAPACITY
        if line.startswith("CAPACITY"):
            vehicle_capacity = int(line.split(":")[1].strip())
            continue

        # Identifier les sections
        if line == "NODE_COORD_SECTION":
            reading_coords = True
            reading_demand = False
            continue
        elif line == "DEMAND_SECTION":
            reading_coords = False
            reading_demand = True
            continue
        elif line == "DEPOT_SECTION":
            break  # Fin des sections pertinentes

        # Lire les coordonnées des nœuds
        if reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                customer_id = int(parts[0])
                x_coord = float(parts[1])
                y_coord = float(parts[2])
                NODE_COORD_SECTION.append((customer_id, (x_coord, y_coord)))
                num_customers += 1
            continue

        # Lire les demandes des clients
        if reading_demand:
            parts = line.split()
            if len(parts) >= 2:
                customer_id = int(parts[0])
                demand = int(parts[1])
                DEMAND_SECTION.append((customer_id, demand))
            continue

    # Assigner la demande à chaque client
    # Assumer que le client 1 est le dépôt avec demande 0
    customer_data = []
    for coord in NODE_COORD_SECTION:
        cid, pos = coord
        demand = 0  # Par défaut
        for dem in DEMAND_SECTION:
            if dem[0] == cid:
                demand = dem[1]
                break
        customer_data.append((cid, pos, demand))

    # Identifier le dépôt (client avec demande 0)
    depot = next((cust for cust in customer_data if cust[0] == 1), None)

    return num_customers, vehicle_capacity, depot, customer_data

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

def create_graph(num_customers, depot, customer_data):
    G = nx.Graph()

    # Ajouter le dépôt
    G.add_node('depot', pos=depot[1], demand=depot[2])

    # Ajouter les clients
    for cust in customer_data:
        cid, pos, demand = cust
        if cid != depot[0]:
            G.add_node(f'customer_{cid}', pos=pos, demand=demand)

    # Ajouter les arêtes avec les distances
    nodes = ['depot'] + [f'customer_{cust[0]}' for cust in customer_data if cust[0] != depot[0]]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            pos_i = G.nodes[node_i]['pos']
            pos_j = G.nodes[node_j]['pos']
            distance = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
            G.add_edge(node_i, node_j, weight=distance)

    return G

def plot_graph_with_solution(graph, solution):
    pos = nx.get_node_attributes(graph, 'pos')

    # Plot nodes without labels and edges
    nx.draw(graph, pos, with_labels=False, node_size=150, node_color='lightblue', edgecolors='white', width =0.1)

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    tour = []
    index = 0
    for id, node in enumerate(solution):
        tour.append(node)
        if id != 0 and node == 'depot':
            # Highlight the solution edges
            solution_edges = [(tour[i], tour[i + 1]) for i in range(len(tour) - 1)]
            nx.draw(graph, pos, edgelist=solution_edges, edge_color=colors[index], width=2, with_labels=False, node_size=150, node_color='lightblue', edgecolors='black', linewidths=0.8)
            index += 1
            tour = [node]

    # Add labels for customers without overlap
    labels = {node: node.split('_')[1] if 'customer' in node else node for node in graph.nodes()}
    pos_lower = {k: (x, y - 0.05) for k, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, pos_lower, labels, font_size=8, font_color='black', verticalalignment="bottom")

    # Add labels for demands
    demand_labels = {node: f'({graph.nodes[node]["demand"]})' for node in graph.nodes() if node != 'depot'}
    pos_higher = {k: (x, y - 0.15) for k, (x, y) in pos.items()}
    nx.draw_networkx_labels(graph, pos_higher, demand_labels, font_size=6, font_color='blue', verticalalignment="bottom")

    plt.title('Vehicle Routing Problem - Solution')
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

def generate_neighbor(solution, demands, capacity):
    # Function to generate a neighbor solution by swapping two nodes in a solution
    new_solution = [route[:] for route in solution]  # Deep copy of current solution
    
    # Check if there are at least two routes to swap between
    if len(new_solution) < 2:
        return solution  # Return the original solution if there are fewer than two routes

    # Select two random routes and swap a node between them if feasible
    route1, route2 = random.sample(range(len(new_solution)), 2)
    if new_solution[route1] and new_solution[route2]:
        node1 = random.choice(new_solution[route1])
        node2 = random.choice(new_solution[route2])
        
        # Extract demands for the selected nodes
        demand_node1 = next(demande for ville, demande in demands if ville == node1)
        demand_node2 = next(demande for ville, demande in demands if ville == node2)
        
        # Check if swapping maintains capacity constraints
        load_route1 = sum(next(demande for ville, demande in demands if ville == node) for node in new_solution[route1]) - demand_node1 + demand_node2
        load_route2 = sum(next(demande for ville, demande in demands if ville == node) for node in new_solution[route2]) - demand_node2 + demand_node1
                
        if load_route1 <= capacity and load_route2 <= capacity:
            # Perform swap
            idx1, idx2 = new_solution[route1].index(node1), new_solution[route2].index(node2)
            new_solution[route1][idx1], new_solution[route2][idx2] = node2, node1
    
    return new_solution

def evaluer_solution(solution, NODE_COORD_SECTION):
    return distance_totale(solution, NODE_COORD_SECTION)

def verifier_capacite(solution, demandes):
    capacites = [0] * len(solution)
    for vehicule, route in enumerate(solution):
        for ville in route:
            demande = next((demande for id_ville, demande in demandes if id_ville == ville), 0)
            capacites[vehicule] += demande
            
            if capacites[vehicule] > capacite_vehicule:
                return False  # Dépassement de capacité pour ce véhicule
    return True

# Displaying the routes and total cost
def afficher_solution(solution, distance):
    for i, vehicule in enumerate(solution, 1):
        if vehicule:  # If the vehicle has assigned cities
            villes = " ".join(map(str, vehicule))
            print(f"Route #{i}: {villes}")
    print(f"Le coût de la meilleure solution: {distance:.4f}")