from Metaheuristics import * 

def test_Hill_Climbing():
    ''' ---------- Hill Climbing ---------- '''
    start_time = time.perf_counter()

    HC = Hill_Climbing('INSTANCES_random\\C-n5-k2.vrp')
    meilleure_solution, meilleure_distance = HC.run() 

    end_time = time.perf_counter()
    
    print(f"Recherche locale terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Tabu_Search():
    ''' ---------- Tabu Search ---------- '''
    start_time = time.perf_counter()

    TS = Tabu_Search('INSTANCES_random\\C-n5-k2.vrp')
    meilleure_solution, meilleure_distance = TS.run() 

    end_time = time.perf_counter()
    
    print(f"Tabu Search terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Simulated_Annealing():
    ''' ---------- Simulated Annealing ---------- '''
    start_time = time.perf_counter()

    SA = Simulated_Annealing('INSTANCES_random\\C-n5-k2.vrp')
    initial_temp = 1000
    cooling_rate = 0.99
    min_temp = 1
    max_iter = 100
    meilleure_solution, meilleure_distance = SA.run(initial_temp, cooling_rate, min_temp, max_iter) 

    end_time = time.perf_counter()
    
    print(f"Simulated Annealing terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Golden_Ball_Algorithm():
    ''' ---------- Golden Ball Algorithm ---------- '''
    start_time = time.perf_counter()

    GBA = Golden_Ball_Algorithm('INSTANCES_random\\C-n5-k2.vrp')
    num_generations = 100
    population_size = 33
    num_teams = 3 
    
    meilleure_solution, meilleure_distance = GBA.run(population_size, num_teams, num_generations) 

    end_time = time.perf_counter()
    
    print(f"Golden Ball Algorithm terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Genetic_Algorithm():
    ''' ---------- Genetic Algorithm ---------- '''
    start_time = time.perf_counter()

    GA = Genetic_Algorithm('INSTANCES_random\\C-n5-k2.vrp')
    taille_population = 150
    generations = 500
    mutation_rate = 0.1
    
    meilleure_solution, meilleure_distance = GA.run(taille_population, generations, mutation_rate) 

    end_time = time.perf_counter()
    
    print(f"Genetic Algorithm terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Partical_Swarm_Optimization():
    ''' ---------- Partical Swarm Opitmization ---------- '''
    start_time = time.perf_counter()

    PSO = Partical_Swarm_Optimization('INSTANCES_random\\C-n5-k2.vrp')
    num_particles =  50       
    num_iterations = 200       
    inertia_weight = 0.4        
    cognitive_weight = 1.5      
    social_weight = 1.7
    
    meilleure_solution, meilleure_distance = PSO.run(num_particles, num_iterations, inertia_weight, cognitive_weight, social_weight) 

    end_time = time.perf_counter()
    
    print(f"Partical Swarm Opitmization terminée.") 
    # Display the final solution
    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def test_Ant_Colony_Optimization():
    ''' ---------- Ant Colony Opitmization ---------- '''
    ACO = Ant_Colony_Optimization('INSTANCES_random\\C-n5-k2.vrp')

    # Résoudre le VRP en utilisant ACO 
    start_time = time.time()
    meilleure_solution, meilleure_distance = ACO.run(num_iterations=100, num_ants=100, alpha=1, beta=3, evaporation_rate=0.2, Q=70)
    end_time = time.time()

    print(f" Ant Colony Opitmization terminée.") 

    afficher_solution(meilleure_solution, meilleure_distance)

    print(f"Temps total d'exécution: {end_time - start_time:.4f} secondes.")

def main():
    test_Hill_Climbing()
    test_Tabu_Search()
    test_Simulated_Annealing()
    test_Golden_Ball_Algorithm()
    test_Genetic_Algorithm()
    test_Partical_Swarm_Optimization()
    test_Ant_Colony_Optimization()

if __name__ == "__main__":
    main()