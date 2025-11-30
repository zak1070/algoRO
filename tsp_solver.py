import sys
import pulp
import time
import itertools

# ==============================================================================
#   FONCTIONS UTILITAIRES (Lecture & Graphes)
# ==============================================================================

def read_instance(filename):
    """Lit le fichier d'instance. Gère les sauts de ligne arbitraires [cite: 33-40]."""
    try:
        with open(filename, 'r') as f:
            content = f.read().split()
    except FileNotFoundError:
        print(f"Erreur: Fichier '{filename}' introuvable.")
        sys.exit(1)
    
    if not content:
        sys.exit(1)

    iterator = iter(content)
    try:
        n = int(next(iterator))
        coords = []
        for _ in range(n):
            coords.append((float(next(iterator)), float(next(iterator))))
            
        dist_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(float(next(iterator)))
            dist_matrix.append(row)
            
        return n, coords, dist_matrix
    except StopIteration:
        sys.exit(1)

def extract_edges(n, x_vars):
    """Récupère les arcs actifs (valeur > 0.9)."""
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                val = pulp.value(x_vars.get((i, j)))
                if val and val > 0.9: 
                    edges.append((i, j))
    return edges

def find_subtours(n, edges):
    """Trouve les sous-tours (composantes connexes)."""
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
    
    visited = set()
    subtours = []
    
    for i in range(n):
        if i not in visited:
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
    """Reconstruit le cycle pour l'affichage."""
    edges = extract_edges(n, x_vars)
    if not edges: return []
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
    tour = [0]
    curr = 0
    while len(tour) < n:
        found_next = False
        for neighbor in adj.get(curr, []):
            if neighbor not in tour:
                tour.append(neighbor)
                curr = neighbor
                found_next = True
                break
        if not found_next: break
    tour.append(0)
    return tour

# ==============================================================================
#   FORMULATIONS (MTZ & DFJ)
# ==============================================================================

def solve_mtz(n, dist_matrix, relax=False):
    # f=0 (Entier) ou f=1 (Relaxation) [cite: 94-95]
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    cat_type = pulp.LpContinuous if relax else pulp.LpBinary
    
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, cat_type)

    u = {}
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", 1, n, pulp.LpContinuous)

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # Contraintes de degré
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    # Contraintes MTZ [cite: 11]
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    # Résolution (Mesure du temps solveur uniquement [cite: 88])
    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

def solve_dfj_enumerative(n, dist_matrix, relax=False):
    # f=2 (Entier) ou f=3 (Relaxation) [cite: 96-97]
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

    # Génération a priori de TOUTES les contraintes de sous-tours
    # Optimisation: on exclut le noeud 0 des sous-ensembles pour éviter les doublons (S vs V\S)
    nodes = range(1, n) 
    for size in range(2, n): 

        for subset in itertools.combinations(nodes, size):
            prob += pulp.lpSum(x[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

# def solve_dfj_iterative(n, dist_matrix):
#     # f=4 (Itératif) [cite: 98]
#     prob = pulp.LpProblem("TSP_DFJ_Iter", pulp.LpMinimize)
#     x = {}
#     for i in range(n):
#         for j in range(n):
#             if i != j:
#                 x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)

#     prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

#     for i in range(n):
#         prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
#     for j in range(n):
#         prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

#     solver = pulp.PULP_CBC_CMD(msg=0)
#     total_solve_time = 0
#     iterations = 0
    
#     while True:
#         # Mesure temps solveur cumulatif [cite: 89]
#         t0 = time.time()
#         prob.solve(solver)
#         total_solve_time += (time.time() - t0)
        
#         if prob.status != pulp.LpStatusOptimal:
#             break
            
#         current_edges = extract_edges(n, x)
#         subtours = find_subtours(n, current_edges)
        
#         # Condition d'arrêt : 1 seul cycle couvrant n villes
#         if len(subtours) == 1 and len(subtours[0]) == n:
#             break
        
#         # Ajout des contraintes (Lazy Constraints) [cite: 29]
#         for S in subtours:
#             prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
        
#         iterations += 1
            
#     return prob, x, total_solve_time, iterations
def solve_dfj_iterative(n, dist_matrix):
    """
    DFJ Itératif avec optimisation BONUS.
    Si exactement 2 sous-tours sont détectés, on n'ajoute qu'une seule contrainte (l'autre est redondante).
    """
    # 1. Création du Problème Maître (Master Problem) sans contraintes de sous-tours
    prob = pulp.LpProblem("TSP_DFJ_Iter", pulp.LpMinimize)
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpBinary)

    # Objectif
    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    # Contraintes de degré
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    solver = pulp.PULP_CBC_CMD(msg=0)
    total_solve_time = 0
    iterations = 0
    
    while True:
        # Mesure temps solveur cumulatif [cite: 100-101]
        t0 = time.time()
        prob.solve(solver)
        total_solve_time += (time.time() - t0)
        
        if prob.status != pulp.LpStatusOptimal:
            break
            
        # Détection de sous-tours (Hors chrono)
        current_edges = extract_edges(n, x)
        subtours = find_subtours(n, current_edges)
        
        # Condition d'arrêt : 1 seul cycle couvrant n villes
        if len(subtours) == 1 and len(subtours[0]) == n:
            break
        
        # --- LOGIQUE BONUS  ---
        # Si exactement 2 sous-tours, on ne coupe que le premier car l'autre contrainte est redondante
        if len(subtours) == 2:
            S = subtours[0]
            prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
        else:
            # Cas normal : on ajoute des contraintes pour TOUS les sous-tours détectés
            for S in subtours:
                prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
        
        iterations += 1
            
    return prob, x, total_solve_time, iterations
# ==============================================================================
#   MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tsp_solver.py <instance_file> <f>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        f_flag = int(sys.argv[2])
    except ValueError:
        print("Erreur: f doit être entier.")
        sys.exit(1)

    n, coords, dist_matrix = read_instance(filename)
    
    prob, x_vars, duration, extra = None, None, 0, 0
    
    # Sélection de la formulation [cite: 94-98]
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
        print("Flag invalide (0-4).")
        sys.exit(1)

    # --- AFFICHAGES OBLIGATOIRES  ---
    
    obj_val = pulp.value(prob.objective)
    
    # 1. Valeur objective
    print(f"Objective: {obj_val}")
    
    # 2. Cycle Hamiltonien (Seulement pour solutions entières f=0, 2, 4)
    if f_flag in [0, 2, 4] and prob.status == pulp.LpStatusOptimal:
        tour = get_ordered_tour(n, x_vars)
        print(f"Tour: {tour}")
    
    # 3. Temps de résolution
    print(f"Time: {duration:.4f}")
    
    # 4. Nombre d'itérations (Pour DFJ Itératif f=4)
    if f_flag == 4:
        print(f"Iterations: {extra}")

    # --- AFFICHAGES TECHNIQUES POUR BENCHMARK.PY ---
    # (Nécessaires pour remplir le fichier results.csv avec les colonnes Vars et Constr)
    print(f"Vars: {len(prob.variables())}")
    print(f"Constraints: {len(prob.constraints)}")