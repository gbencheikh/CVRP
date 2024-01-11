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
    