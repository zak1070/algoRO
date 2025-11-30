import os
import subprocess
import csv
import re
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SOLVER_SCRIPT = "tsp_solver.py"
INSTANCE_DIR = "instances"
OUTPUT_CSV = "results.csv"
TIMEOUT = 600  # 10 minutes max par instance
MAX_N_ENUM = 15  # Limite pour DFJ Enum (f=2/3) car trop lent après

def run_solver(filepath, flag):
    """
    Lance tsp_solver.py avec le fichier et le flag donnés.
    Capture et parse la sortie (Objective, Time, Vars, Constraints, Tour).
    """
    try:
        # Appel commande: python3 tsp_solver.py <fichier> <flag>
        cmd = [sys.executable, SOLVER_SCRIPT, filepath, str(flag)]
        
        # Exécution avec timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        
        if result.returncode != 0:
            return None
        
        output = result.stdout
        data = {}
        
        # --- PARSING REGEX ---
        # Ces regex correspondent exactement aux print() de ton tsp_solver.py
        
        # "Objective: 245.3"
        obj = re.search(r"Objective:\s*([\d\.]+)", output)
        if obj: data['obj'] = float(obj.group(1))
        
        # "Time: 0.12"
        tm = re.search(r"Time:\s*([\d\.]+)", output)
        if tm: data['time'] = float(tm.group(1))
        
        # "Vars: 100"
        vr = re.search(r"Vars:\s*(\d+)", output)
        data['vars'] = int(vr.group(1)) if vr else 0
        
        # "Constraints: 190"
        cn = re.search(r"Constraints:\s*(\d+)", output)
        data['constr'] = int(cn.group(1)) if cn else 0
        
        # "Tour: [0, 1, 3, 2, 0]"
        tr = re.search(r"Tour:\s*(\[.*\])", output)
        data['tour'] = tr.group(1) if tr else ""
        
        return data

    except Exception as e:
        print(f"Error running {filepath} f={flag}: {e}")
        return None

def main():
    # Vérification du dossier instances
    if not os.path.exists(INSTANCE_DIR):
        print(f"Erreur: Dossier '{INSTANCE_DIR}' introuvable!")
        return

    # En-têtes exacts demandés par l'image
    headers = ["instance", "formulation", "obj_int", "time_int", "obj_relax", "time_relax", "gap", "vars", "constr"]
    
    # Récupération des fichiers triés
    files = sorted([f for f in os.listdir(INSTANCE_DIR) if f.endswith(".txt")])
    
    # Affichage de l'en-tête dans le terminal
    print(f"{'INSTANCE':<30} {'FORMULATION':<10} {'OBJ_INT':<10} {'TIME':<8} {'GAP':<8} | {'CHEMIN TROUVÉ'}")
    print("-" * 110)

    # Ouverture du fichier CSV en écriture
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for filename in files:
            path = os.path.join(INSTANCE_DIR, filename)
            
            # Extraction de la taille n (ex: instance_10_...)
            try:
                n = int(filename.split('_')[1])
            except:
                n = 999

            # ====================================================
            # 1. FORMULATION MTZ
            # ====================================================
            # f=0: Entier, f=1: Relaxation
            res_int = run_solver(path, 0)
            res_rel = run_solver(path, 1)
            
            if res_int and res_rel:
                # [cite_start]Calcul Gap: (Int - Relax) / Int [cite: 75]
                gap = (res_int['obj'] - res_rel['obj']) / res_int['obj'] if res_int['obj'] != 0 else 0.0
                
                # Print Terminal (AVEC le Tour pour toi)
                print(f"{filename:<30} {'MTZ':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.4f} {gap:<8.4f} | {res_int['tour']}", flush=True)
                
                # Ecriture CSV (SANS le Tour, conforme à l'image)
                writer.writerow({
                    "instance": filename,
                    "formulation": "MTZ",
                    "obj_int": res_int['obj'],
                    "time_int": res_int['time'],
                    "obj_relax": res_rel['obj'],
                    "time_relax": res_rel['time'],
                    "gap": round(gap, 4),
                    "vars": res_int['vars'],
                    "constr": res_int['constr']
                })

            # ====================================================
            # 2. FORMULATION DFJ Enumératif
            # ====================================================
            # f=2: Entier, f=3: Relaxation
            # Uniquement si n <= 15 (sinon trop long)
            if n <= MAX_N_ENUM:
                res_int = run_solver(path, 2)
                res_rel = run_solver(path, 3)
                
                if res_int and res_rel:
                    gap = (res_int['obj'] - res_rel['obj']) / res_int['obj'] if res_int['obj'] != 0 else 0.0
                    
                    print(f"{filename:<30} {'DFJ_enum':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.4f} {gap:<8.4f} | {res_int['tour']}", flush=True)
                    
                    writer.writerow({
                        "instance": filename,
                        "formulation": "DFJ_enum",
                        "obj_int": res_int['obj'],
                        "time_int": res_int['time'],
                        "obj_relax": res_rel['obj'],
                        "time_relax": res_rel['time'],
                        "gap": round(gap, 4),
                        "vars": res_int['vars'],
                        "constr": res_int['constr']
                    })

            # ====================================================
            # 3. FORMULATION DFJ Itératif
            # ====================================================
            # f=4: Entier uniquement
            # Pas de relaxation demandée pour l'itératif
            res_int = run_solver(path, 4)
            
            if res_int:
                print(f"{filename:<30} {'DFJ_iter':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.4f} {'-':<8} | {res_int['tour']}", flush=True)
                
                writer.writerow({
                    "instance": filename,
                    "formulation": "DFJ_iter",
                    "obj_int": res_int['obj'],
                    "time_int": res_int['time'],
                    "obj_relax": "",  # Non applicable
                    "time_relax": "", # Non applicable
                    "gap": "",        # Non applicable
                    "vars": res_int['vars'],
                    "constr": res_int['constr'] # Nombre final de contraintes
                })

    print("-" * 110)
    print(f"Terminé ! Les résultats sont dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()