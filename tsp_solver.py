import sys
import pulp
import time
import itertools

# ==============================================================================
#   1. LECTURE & PARSING DE L'INSTANCE
# ==============================================================================

def read_instance(filename):
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

# ==============================================================================
#   2. OUTILS DE GRAPHES
# ==============================================================================

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
#   3. FORMULATIONS (MTZ & DFJ)
# ==============================================================================

def solve_mtz(n, dist_matrix, relax=False):
    """MTZ: f=0 (Entier) ou f=1 (Relaxation)"""
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

    # Contraintes MTZ
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

def solve_dfj_enumerative(n, dist_matrix, relax=False):
    """DFJ Énumératif: f=2 (Entier) ou f=3 (Relaxation)"""
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

    # Génération a priori (Optimisation: on exclut le noeud 0)
    nodes = range(1, n) 
    for size in range(2, n): 
        for subset in itertools.combinations(nodes, size):
            prob += pulp.lpSum(x[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

def solve_dfj_iterative(n, dist_matrix, bonus_mode=False):
    """
    DFJ Itératif: f=4 (Standard) ou f=5 (Bonus)
    bonus_mode = True : Si 2 sous-tours, on ajoute une seule contrainte.
    bonus_mode = False : On ajoute toutes les contraintes.
    """
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
        # Mesure temps solveur cumulatif
        t0 = time.time()
        prob.solve(solver)
        total_solve_time += (time.time() - t0)
        
        if prob.status != pulp.LpStatusOptimal:
            break
            
        current_edges = extract_edges(n, x)
        subtours = find_subtours(n, current_edges)
        
        if len(subtours) == 1 and len(subtours[0]) == n:
            break
        
        # --- LOGIQUE DE COUPE ---
        # Si Bonus activé ET exactement 2 sous-tours -> 1 seule contrainte
        if bonus_mode and len(subtours) == 2:
            S = subtours[0]
            prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
        else:
            # Sinon (Standard ou >2 sous-tours) -> Toutes les contraintes
            for S in subtours:
                prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
        
        iterations += 1
            
    return prob, x, total_solve_time, iterations

# ==============================================================================
#   4. MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tsp_solver.py <instance_file> <f>")
        sys.exit(1)

    filename = sys.argv[1]
    try:
        f_flag = int(sys.argv[2])
    except ValueError:
        print("Erreur: f doit être un entier.")
        sys.exit(1)

    n, coords, dist_matrix = read_instance(filename)
    
    prob, x_vars, duration, extra = None, None, 0, 0
    
    # Sélection de la formulation
    if f_flag == 0:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=False)
    elif f_flag == 1:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=True)
    elif f_flag == 2:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=False)
    elif f_flag == 3:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=True)
    elif f_flag == 4:
        # Standard
        prob, x_vars, duration, extra = solve_dfj_iterative(n, dist_matrix, bonus_mode=False)
    elif f_flag == 5:
        # Bonus (Test expérimental)
        prob, x_vars, duration, extra = solve_dfj_iterative(n, dist_matrix, bonus_mode=True)
    else:
        print("Flag invalide (0-5).")
        sys.exit(1)

    # --- AFFICHAGES ---
    
    obj_val = pulp.value(prob.objective)
    
    print(f"Objective: {obj_val}")
    print(f"Time: {duration:.4f}")
    
    if f_flag in [4, 5]:
        print(f"Iterations: {extra}")

    if f_flag in [0, 2, 4, 5] and prob.status == pulp.LpStatusOptimal:
        tour = get_ordered_tour(n, x_vars)
        print(f"Tour: {tour}")

    # Pour benchmark.py (CSV)
    print(f"Vars: {len(prob.variables())}")
    print(f"Constraints: {len(prob.constraints)}")