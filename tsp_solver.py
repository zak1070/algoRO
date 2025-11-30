import sys
import pulp
import time
import itertools

# ==============================================================================
#  LECTURE DE L'INSTANCE
# ==============================================================================

def read_instance(filename):
    """
    Lit le fichier d'instance selon le format spécifié [cite: 33-40].
    Gère les sauts de ligne arbitraires via un itérateur.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read().split()
    except FileNotFoundError:
        print(f"Erreur: Fichier '{filename}' introuvable.")
        sys.exit(1)
    
    if not content:
        print("Erreur: Fichier vide.")
        sys.exit(1)

    iterator = iter(content)
    try:
        n = int(next(iterator))
        coords = []
        # Lecture des coordonnées
        for _ in range(n):
            coords.append((float(next(iterator)), float(next(iterator))))
            
        dist_matrix = []
        # Lecture de la matrice de distances
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(next(iterator)))
            dist_matrix.append(row)
            
        return n, coords, dist_matrix
    except StopIteration:
        print("Erreur : Fichier d'instance incomplet ou mal formaté.")
        sys.exit(1)

# ==============================================================================
#  OUTILS DE GRAPHES & AFFICHAGE
# ==============================================================================

def extract_edges(n, x_vars):
    """Récupère les arcs actifs (valeur proche de 1)."""
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                # Tolérance pour les erreurs d'arrondi flottant
                val = pulp.value(x_vars.get((i, j)))
                if val and val > 0.9: 
                    edges.append((i, j))
    return edges

def find_subtours(n, edges):
    """
    Identifie les composantes connexes (sous-tours) dans la solution courante.
    Retourne une liste de listes de sommets.
    """
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
    
    visited = set()
    subtours = []
    
    for i in range(n):
        if i not in visited:
            # Si le sommet n'est pas connecté (cas rare en relaxation), on l'ignore ou on le traite seul
            if not adj[i] and i not in [u for u,v in edges] and i not in [v for u,v in edges]:
                continue

            component = []
            stack = [i]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        stack.append(neighbor)
            subtours.append(component)
    return subtours

def get_ordered_tour(n, x_vars):
    """Reconstruit le cycle hamiltonien ordonné pour l'affichage[cite: 101]."""
    edges = extract_edges(n, x_vars)
    if not edges:
        return []
    
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        
    tour = [0]
    curr = 0
    # On parcourt le cycle
    while len(tour) < n:
        found_next = False
        for neighbor in adj.get(curr, []):
            if neighbor not in tour:
                tour.append(neighbor)
                curr = neighbor
                found_next = True
                break
        if not found_next:
            break
    # Fermer le cycle pour l'affichage
    tour.append(0)
    return tour

# ==============================================================================
#  FORMULATIONS (MTZ & DFJ)
# ==============================================================================

def solve_mtz(n, dist_matrix, relax=False):
    """
    MTZ (Miller-Tucker-Zemlin).
    f=0 (relax=False) ou f=1 (relax=True).
    """
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    
    # Variables de décision
    # Si relax=True, les variables deviennent continues [0,1] [cite: 55]
    cat_type = pulp.LpContinuous if relax else pulp.LpBinary
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat_type)

    # Variables auxiliaires u_i (pour i=1..n-1)
    u = {}
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", 1, n, pulp.LpContinuous)

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # Contraintes d'assignation (degré)
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    # Contraintes MTZ : u_i - u_j + n*x_ij <= n-1
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    # Mesure du temps (Solveur uniquement)
    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

def solve_dfj_enumerative(n, dist_matrix, relax=False):
    """
    DFJ Énumératif. Génère TOUTES les contraintes a priori.
    f=2 (relax=False) ou f=3 (relax=True).
    """
    prob = pulp.LpProblem("TSP_DFJ_Enum", pulp.LpMinimize)
    
    cat_type = pulp.LpContinuous if relax else pulp.LpBinary
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat_type)

    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    # Génération des contraintes de sous-tours
    # On génère tous les sous-ensembles S de taille 2 à n-1.
    # Optimisation : On ne considère que les sous-ensembles ne contenant pas le sommet 0.
    # Cela évite la redondance (S vs V\S) et divise par 2 le nombre de contraintes (O(2^n)).
    nodes = range(1, n) 
    count_constraints = 0
    
    # Génération Python (Non chronométrée pour respecter [cite: 119])
    for size in range(2, n): 
        for subset in itertools.combinations(nodes, size):
            # sum(x_ij for i,j in S) <= |S| - 1
            prob += pulp.lpSum(x[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1
            count_constraints += 1

    # Mesure du temps (Solveur uniquement)
    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, count_constraints

def solve_dfj_iterative(n, dist_matrix):
    """
    DFJ Itératif. Génération de contraintes (Row Generation).
    f=4.
    """
    # Problème Maître (Master Problem)
    prob = pulp.LpProblem("TSP_DFJ_Iter", pulp.LpMinimize)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)

    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    solver = pulp.PULP_CBC_CMD(msg=0)
    total_solve_time = 0
    iterations = 0
    
    while True:
        iterations += 1
        
        # Mesure du temps : UNIQUEMENT l'appel au solveur [cite: 89, 120]
        t0 = time.time()
        prob.solve(solver)
        total_solve_time += (time.time() - t0)
        
        if prob.status != pulp.LpStatusOptimal:
            break
            
        # Détection de sous-tours (Hors chrono)
        current_edges = extract_edges(n, x)
        subtours = find_subtours(n, current_edges)
        
        # Condition d'arrêt : Un seul cycle couvrant tous les sommets
        if len(subtours) == 1 and len(subtours[0]) == n:
            break
        
        # Ajout des contraintes (cuts) pour chaque sous-tour détecté
        for S in subtours:
            prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
            
    return prob, x, total_solve_time, iterations

# ==============================================================================
#  MAIN
# ==============================================================================

if __name__ == "__main__":
    # Vérification arguments
    if len(sys.argv) < 3:
        print("Usage: python3 tsp_solver.py <instance_file> <f>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        f_flag = int(sys.argv[2])
    except ValueError:
        print("Erreur: le paramètre f doit être un entier.")
        sys.exit(1)

    # Lecture
    n, coords, dist_matrix = read_instance(filename)
    
    # Initialisation résultats
    prob, x_vars, duration, extra = None, None, 0, 0
    
    # Sélecteur de formulation [cite: 94-98]
    if f_flag == 0:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=False)
    elif f_flag == 1:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=True)
    elif f_flag == 2:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=False)
    elif f_flag == 3:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=True)
    elif f_flag == 4:
        prob, x_vars, duration, extra = solve_dfj_iterative(n, dist_matrix)
    else:
        print("Erreur: flag f non reconnu (0-4).")
        sys.exit(1)

    # Affichage des résultats [cite: 99-103]
    # Note: On utilise pulp.value() pour récupérer la valeur objective
    obj_val = pulp.value(prob.objective)
    
    print(f"Objective: {obj_val}")
    print(f"Time: {duration:.4f}") # En secondes
    
    if f_flag == 4:
        print(f"Iterations: {extra}")

    # Affichage du cycle hamiltonien (Uniquement pour les solutions entières)
    # Les relaxations (1, 3) ne produisent pas nécessairement un cycle valide.
    if f_flag in [0, 2, 4] and prob.status == pulp.LpStatusOptimal:
        tour = get_ordered_tour(n, x_vars)
        print(f"Tour: {tour}")