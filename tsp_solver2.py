import sys
import time
import pulp
from itertools import combinations

# ------------------------------------------------------------
# Fonctions Utilitaires
# ------------------------------------------------------------

def read_instance(filename):
    """
    Lit un fichier d'instance TSP et retourne n, les coordonnées et la matrice de coûts.
    """
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    V = range(n)

    coords = []
    for i in V:
        x_str, y_str = lines[1 + i].split()
        coords.append((float(x_str), float(y_str)))

    dist = []
    start = 1 + n
    for i in V:
        row = [float(x) for x in lines[start + i].split()]
        dist.append(row)

    return n, coords, dist

def reconstruct_tour(n, V, x):
    """
    Reconstruit le cycle hamiltonien à partir des valeurs des variables x_ij.
    """
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        next_city = None
        for j in V:
            if j != current and pulp.value(x[current][j]) > 0.5:
                next_city = j
                break
        
        if next_city is None:
            return tour if len(tour) == n else None

        tour.append(next_city)
        current = next_city
    
    if pulp.value(x[current][0]) > 0.5:
        tour.append(0)
    else:
         return None

    return tour


def create_tsp_base_model(n, dist, relax, model_name):
    """
    Crée le modèle de base du TSP (objectif et contraintes de degré).
    """
    V = range(n)
    prob = pulp.LpProblem(model_name, pulp.LpMinimize)

    x_cat = pulp.LpBinary if not relax else pulp.LpContinuous
    x = pulp.LpVariable.dicts(
        "x",
        (V, V),
        lowBound=0,
        upBound=1,
        cat=x_cat
    )

    prob += pulp.lpSum(dist[i][j] * x[i][j] for i in V for j in V), "Total_Cost"

    for i in V:
        prob += x[i][i] == 0, f"No_Self_Loop_{i}"

    for i in V:
        prob += pulp.lpSum(x[i][j] for j in V) == 1, f"Out_Degree_{i}"
        prob += pulp.lpSum(x[j][i] for j in V) == 1, f"In_Degree_{i}"

    return prob, x, V

def find_cycles(n, x):
    """
    Détecte tous les cycles (sous-tours) dans la solution actuelle X.
    Retourne une liste de sous-tours, où chaque sous-tour est un tuple d'indices de ville.
    """
    V = range(n)
    
    # 1. Construire la liste des successeurs à partir des variables x_ij actives
    successors = {}
    for i in V:
        for j in V:
            if pulp.value(x[i][j]) > 0.5:
                successors[i] = j
                break

    # 2. Initialisation
    unvisited = set(V)
    cycles = []

    # 3. Parcours des nœuds non visités
    while unvisited:
        start_node = unvisited.pop() 
        
        path = [start_node]
        current_node = start_node
        
        while current_node in successors:
            current_node = successors[current_node]
            
            if current_node in path:
                start_index = path.index(current_node)
                cycle = tuple(path[start_index:])
                
                if len(cycle) >= 2:
                    cycles.append(cycle)
                    
                    for node in cycle:
                        unvisited.discard(node)
                break
            
            path.append(current_node)
            unvisited.discard(current_node) 
            
            if len(path) > n:
                print("Avertissement: Problème dans la détection de cycles.")
                break
                
    return cycles

# ------------------------------------------------------------
# Formulation MTZ (f = 0, f = 1)
# ------------------------------------------------------------
def solve_mtz(n, dist, relax=False):
    """
    Résout le TSP avec la formulation MTZ (Miller-Tucker-Zemlin).
    """
    prob, x, V = create_tsp_base_model(n, dist, relax, "TSP_MTZ")

    # Variables u_i (pour l'ordre de visite et l'élimination des sous-tours)
    u = pulp.LpVariable.dicts(
        "u",
        V,
        lowBound=0,
        upBound=n - 1,
        cat=pulp.LpContinuous
    )

    prob += u[0] == 0, "Reference_City_U0"

    # Contraintes MTZ pour éliminer les sous-tours :
    for i in V:
        for j in V:
            if i != 0 and j != 0 and i != j: 
                prob += u[i] - u[j] + n * x[i][j] <= n - 1, f"MTZ_Subtour_Elimination_{i}_{j}"

    start_time = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.perf_counter() - start_time

    obj_value = pulp.value(prob.objective)
    
    tour = None
    if not relax and prob.status == pulp.LpStatusOptimal:
        tour = reconstruct_tour(n, V, x)

    return tour, obj_value, solve_time, prob.numConstraints()

# ------------------------------------------------------------
# Formulation DFJ énumérative (f = 2, f = 3)
# ------------------------------------------------------------
def solve_dfj_enum(n, dist, relax=False):
    """
    Résout le TSP avec la formulation DFJ (Dantzig-Fulkerson-Johnson) énumérative.
    """
    prob, x, V = create_tsp_base_model(n, dist, relax, "TSP_DFJ_ENUM")

    # Contraintes DFJ énumératives pour éliminer les sous-tours :
    for k in range(2, n):
        for S in combinations(V, k):
            prob += (
                pulp.lpSum(x[i][j] for i in S for j in S if j != i) <= k - 1, 
                f"DFJ_Subtour_Elimination_Size_{k}_{'_'.join(map(str, S))}"
            )

    start_time = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    solve_time = time.perf_counter() - start_time

    obj_value = pulp.value(prob.objective)
    
    tour = None
    if not relax and prob.status == pulp.LpStatusOptimal:
        tour = reconstruct_tour(n, V, x)

    total_constraints = prob.numConstraints() 
    
    return tour, obj_value, solve_time, total_constraints

# ------------------------------------------------------------
# Formulation DFJ itérative (f = 4, f = 5 [Bonus])
# ------------------------------------------------------------
def solve_dfj_iter(n, dist, run_bonus=False):
    """
    Résout le TSP avec la formulation DFJ avec génération itérative de contraintes.
    Si run_bonus est True, n'ajoute qu'une seule coupe si exactement 2 sous-tours sont trouvés.
    """
    prob, x, V = create_tsp_base_model(n, dist, relax=False, model_name="TSP_DFJ_ITER")
    
    num_iterations = 0
    total_solve_time = 0.0
    
    while True:
        num_iterations += 1
        
        # Résoudre le modèle actuel (mesurer seulement le temps du solveur)
        start_time = time.perf_counter()
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        total_solve_time += time.perf_counter() - start_time
        
        if prob.status != pulp.LpStatusOptimal:
            print(f"Erreur de solveur à l'itération {num_iterations}: {pulp.LpStatus[prob.status]}")
            break
        
        cycles = find_cycles(n, x)
        
        # Condition d'arrêt : Solution optimale trouvée (un seul cycle visitant n villes)
        if len(cycles) == 1 and len(cycles[0]) == n:
            break
        
        # Sinon, il y a des sous-tours, on ajoute des contraintes
        else:
            cycles_to_cut = cycles # Par défaut, on ajoute toutes les coupes trouvées
            
            # --- LOGIQUE DU BONUS ---
            if run_bonus and len(cycles) == 2:
                # Si le bonus est activé et qu'il y a exactement 2 sous-tours,
                # on n'ajoute qu'une seule coupe (la première), car l'autre est redondante.
                cycles_to_cut = cycles[:1] 
                print(f"[BONUS ACTIVÉ] 2 sous-tours détectés. Ajout d'une seule contrainte pour {cycles_to_cut[0]}.")
            # ------------------------

            # Ajout des contraintes DFJ pour les cycles sélectionnés
            for S in cycles_to_cut:
                k = len(S)
                
                prob += (
                    pulp.lpSum(x[i][j] for i in S for j in S if i != j) <= k - 1,
                    f"DFJ_Cut_Iter_{num_iterations}_Size_{k}_{'_'.join(map(str, S))}"
                )
        
    obj_value = pulp.value(prob.objective)
    tour = reconstruct_tour(n, V, x)
    
    total_constraints = prob.numConstraints()
    
    return tour, obj_value, total_solve_time, total_constraints, num_iterations

# ------------------------------------------------------------
# Point d'entrée principal du programme
# ------------------------------------------------------------
def main():
    # Ajout du cas f=5 pour le bonus
    if len(sys.argv) != 3:
        print("Usage : python3 tsp_solver.py <instance_file> <f>")
        print("f=0: MTZ entier | f=1: MTZ relaxé | f=2: DFJ enum entier | f=3: DFJ enum relaxé")
        print("f=4: DFJ itératif standard | f=5: DFJ itératif BONUS (single cut pour 2 sous-tours)")
        sys.exit(1)

    instance_file = sys.argv[1]
    try:
        f = int(sys.argv[2])
    except ValueError:
        print("Erreur: Le paramètre 'f' doit être un entier.")
        sys.exit(1)

    n, coords, dist = read_instance(instance_file)
    print(f"\n--- Instance chargée : {instance_file} ({n} villes) ---\n")

    # Variables de sortie
    tour = None
    obj = None
    solve_time = None
    num_constraints = None
    num_iterations = None

    # L'argument run_bonus est False par défaut pour tous les cas sauf f=5
    run_bonus_flag = False

    if f == 0:
        tour, obj, solve_time, num_constraints = solve_mtz(n, dist, relax=False)
        print("=== Formulation MTZ (Entière) ===")
    
    elif f == 1:
        tour, obj, solve_time, num_constraints = solve_mtz(n, dist, relax=True)
        print("=== Formulation MTZ (Relaxation Continue) ===")

    elif f == 2:
        tour, obj, solve_time, num_constraints = solve_dfj_enum(n, dist, relax=False)
        print("=== Formulation DFJ Énumérative (Entière) ===")

    elif f == 3:
        tour, obj, solve_time, num_constraints = solve_dfj_enum(n, dist, relax=True)
        print("=== Formulation DFJ Énumérative (Relaxation Continue) ===")

    elif f == 4:
        tour, obj, solve_time, num_constraints, num_iterations = solve_dfj_iter(n, dist, run_bonus=False)
        print("=== Formulation DFJ Itérative (Entière) - Standard ===")
        
    elif f == 5:
        run_bonus_flag = True # Active le mode bonus
        tour, obj, solve_time, num_constraints, num_iterations = solve_dfj_iter(n, dist, run_bonus=True)
        print("=== Formulation DFJ Itérative (Entière) - BONUS (Single Cut) ===")
    
    else:
        print(f"Erreur: Le paramètre f = {f} n'est pas une formulation valide.")
        sys.exit(1)

    # Affichage des résultats
    print(f"Valeur objective : {obj}")
    print(f"Temps solveur    : {solve_time:.4f} secondes")
    
    # Calcul du nombre de variables : n^2 pour DFJ, n^2 + n pour MTZ
    num_vars = n*n + (n if f in [0, 1] else 0)
    print(f"Nombre de variables : {num_vars}")
    print(f"Nombre de contraintes : {num_constraints}")
    
    if f in [4, 5]:
        print(f"Nombre d'itérations : {num_iterations}")

    # Affichage conditionnel du cycle
    if tour is not None and len(tour) == n + 1:
        print(f"Cycle trouvé    : {tour}")
    elif f in [1, 3]:
        print("(Pas de cycle à afficher pour la relaxation continue)")
    elif tour is None:
        print("Cycle trouvé    : Non trouvé ou solution non optimale/entière.")


if __name__ == "__main__":
    main()