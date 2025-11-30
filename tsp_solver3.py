import sys
import pulp
import time
import itertools

# ==============================================================================
#   LECTURE DE L'INSTANCE
# ==============================================================================

def read_instance(filename):
    """Lit le fichier d'instance. Gère les sauts de ligne arbitraires."""
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
        print("Erreur : Fichier d'instance incomplet.")
        sys.exit(1)

# ==============================================================================
#   OUTILS DE GRAPHES
# ==============================================================================

def extract_edges(n, x_vars):
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                val = pulp.value(x_vars.get((i, j)))
                if val and val > 0.9: 
                    edges.append((i, j))
    return edges

def find_subtours(n, edges):
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
#   FORMULATIONS
# ==============================================================================

def solve_mtz(n, dist_matrix, relax=False):
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

    prob += pulp.lpSum(dist_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j)

    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in range(n) if i != j) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if i != j) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[(i, j)] <= n - 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, 0

def solve_dfj_enumerative(n, dist_matrix, relax=False):
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

    nodes = range(1, n) 
    count_constraints = 0
    # Optimisation Code A : on fixe le noeud 0, donc on itère sur range(1, n)
    for size in range(2, n): 
        for subset in itertools.combinations(nodes, size):
            prob += pulp.lpSum(x[(i, j)] for i in subset for j in subset if i != j) <= len(subset) - 1
            count_constraints += 1

    start_t = time.time()
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duration = time.time() - start_t
    
    return prob, x, duration, count_constraints

def solve_dfj_iterative(n, dist_matrix, bonus_mode=False):
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
        t0 = time.time()
        prob.solve(solver)
        total_solve_time += (time.time() - t0)
        
        if prob.status != pulp.LpStatusOptimal:
            break
            
        current_edges = extract_edges(n, x)
        subtours = find_subtours(n, current_edges)
        
        if len(subtours) == 1 and len(subtours[0]) == n:
            break
        
        # Logique des coupes
        cycles_to_cut = subtours
        if bonus_mode and len(subtours) == 2:
            cycles_to_cut = [subtours[0]] # On ne coupe que le premier
            
        for S in cycles_to_cut:
            prob += pulp.lpSum(x[(i, j)] for i in S for j in S if i != j) <= len(S) - 1
            
    return prob, x, total_solve_time, iterations

# ==============================================================================
#   MAIN
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tsp_solver.py <instance> <f>")
        sys.exit(1)

    filename = sys.argv[1]
    f_flag = int(sys.argv[2])

    n, coords, dist_matrix = read_instance(filename)
    
    prob, x_vars, duration, extra = None, None, 0, 0
    
    if f_flag == 0:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=False)
    elif f_flag == 1:
        prob, x_vars, duration, _ = solve_mtz(n, dist_matrix, relax=True)
    elif f_flag == 2:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=False)
    elif f_flag == 3:
        prob, x_vars, duration, _ = solve_dfj_enumerative(n, dist_matrix, relax=True)
    elif f_flag == 4:
        prob, x_vars, duration, extra = solve_dfj_iterative(n, dist_matrix, bonus_mode=False)
    elif f_flag == 5: # MODE BONUS
        prob, x_vars, duration, extra = solve_dfj_iterative(n, dist_matrix, bonus_mode=True)

    obj_val = pulp.value(prob.objective)
    
    # --- PRINTS POUR CSV PARSING ---
    print(f"Objective: {obj_val}")
    print(f"Time: {duration:.4f}")
    
    # Nombre de variables (Varie selon relax ou entier)
    print(f"Vars: {len(prob.variables())}")
    
    # Nombre de contraintes
    # Pour DFJ Enum (f=2/3), prob.constraints contient tout.
    # Pour DFJ Iter (f=4/5), prob.constraints contient le final après itérations.
    print(f"Constraints: {len(prob.constraints)}")

    if f_flag in [4, 5]:
        print(f"Iterations: {extra}")

    if f_flag in [0, 2, 4, 5] and prob.status == pulp.LpStatusOptimal:
        tour = get_ordered_tour(n, x_vars)
        print(f"Tour: {tour}")