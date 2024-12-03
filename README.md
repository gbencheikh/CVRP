# Metaheuristics for CVRP

## Project Description
This project implements various metaheuristics to solve the Vehicle Routing Problem (CVRP). The CVRP is a combinatorial optimization problem where the goal is to determine optimal routes for a fleet of vehicles to serve a set of customers with specific demands while respecting the vehicles' capacity constraints.

The implemented algorithms allow for performance comparisons on different CVRP instances. Each method aims to minimize a cost function, typically defined as the total distance traveled, while ensuring all constraints are met.

## Implemented Metaheuristics
The project includes implementations of the following metaheuristics:

* Tabu Search (TS)
A local search algorithm that uses a tabu list to avoid cycling and efficiently explore the solution space.

* Hill Climbing (HC)
A simple local search approach that moves to a neighboring solution if it improves the current score.

* Global Ball Optimization (GBO)
A nature-inspired method focused on global optimization principles.

* Particle Swarm Optimization (PSO)
A population-based optimization technique inspired by the social behavior of birds and fish.

* Ant Colony Optimization (ACO)
An algorithm inspired by the foraging behavior of ants, which constructs solutions based on pheromone trails and heuristic information.

* Genetic Algorithm (GA)
An evolutionary approach that combines selection, crossover, and mutation to evolve better solutions over generations.

* Simulated Annealing (SA)
An optimization technique inspired by the annealing process in metallurgy, balancing exploration and exploitation through a temperature-based mechanism.

# Usage
Python 3.8 or later
Required libraries:
numpy
matplotlib
random
Any other dependencies specified in requirements.txt

## Instructions: 

1- Set Up the Environment:
Install dependencies using the following command:

``
!pip install -r requirements.txt
``

2- Run the Project:
Use the main script to select and execute a metaheuristic:

``
python main.py
``

### Customize Parameters:
Modify configuration files or parameters in the script to define the VRP instance and metaheuristic settings.

### Results
Each algorithm outputs:

* The optimized routes for all vehicles.
* The total distance traveled.
* Execution time and convergence metrics.
Comparison plots and detailed performance metrics are generated for analysis.

# Future Work
Implement hybrid metaheuristics for better performance.
Add support for dynamic and stochastic VRP instances.
Integrate real-world data for testing and benchmarking.

# Contributors
This project was developed by Manal and Ghita to explore and compare metaheuristic techniques for combinatorial optimization problems.

Feel free to contribute or suggest improvements!