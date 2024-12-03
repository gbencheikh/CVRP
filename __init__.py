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

def main():
    test_Hill_Climbing()
    test_Tabu_Search()
    test_Simulated_Annealing()

if __name__ == "__main__":
    main()