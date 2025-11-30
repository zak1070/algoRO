import os
import subprocess
import csv
import re
import sys
import time

# CONFIGURATION
SOLVER_SCRIPT = "tsp_solver.py"
INSTANCE_DIR = "instances"
OUTPUT_CSV = "results.csv"
TIMEOUT = 600 # 10 minutes max par run
MAX_N_ENUM = 15 # Limite pour DFJ Enum

def run_solver(filepath, flag):
    """Lance tsp_solver.py et capture la sortie."""
    try:
        cmd = [sys.executable, SOLVER_SCRIPT, filepath, str(flag)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        
        if result.returncode != 0:
            return None
        
        output = result.stdout
        data = {}
        
        # --- PARSING ---
        obj = re.search(r"Objective:\s*([\d\.]+)", output)
        if obj: data['obj'] = float(obj.group(1))
        
        tm = re.search(r"Time:\s*([\d\.]+)", output)
        if tm: data['time'] = float(tm.group(1))
        
        vr = re.search(r"Vars:\s*(\d+)", output)
        data['vars'] = int(vr.group(1)) if vr else 0
        
        cn = re.search(r"Constraints:\s*(\d+)", output)
        data['constr'] = int(cn.group(1)) if cn else 0
        
        tr = re.search(r"Tour:\s*(\[.*\])", output)
        data['tour'] = tr.group(1) if tr else ""
        
        return data

    except Exception as e:
        print(f"Error running {filepath} f={flag}: {e}")
        return None

def main():
    if not os.path.exists(INSTANCE_DIR):
        print(f"Dossier '{INSTANCE_DIR}' introuvable!")
        return

    headers = ["instance", "formulation", "obj_int", "time_int", "obj_relax", "time_relax", "gap", "vars", "constr"]
    
    files = sorted([f for f in os.listdir(INSTANCE_DIR) if f.endswith(".txt")])
    
    print(f"{'INSTANCE':<25} {'FORMULATION':<10} {'OBJ_INT':<10} {'TIME':<8} {'GAP':<8} | {'CHEMIN TROUVÉ'}")
    print("-" * 100)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for filename in files:
            path = os.path.join(INSTANCE_DIR, filename)
            try:
                n = int(filename.split('_')[1])
            except:
                n = 999

            # --- 1. MTZ (Entier + Relaxation) ---
            # On lance f=0 (Entier) et f=1 (Relaxation)
            res_int = run_solver(path, 0)
            res_rel = run_solver(path, 1)
            
            if res_int and res_rel:
                # Calcul du Gap
                gap = (res_int['obj'] - res_rel['obj']) / res_int['obj'] if res_int['obj'] else 0
                
                print(f"{filename:<25} {'MTZ':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.2f} {gap:<8.4f} | {res_int['tour']}")
                
                writer.writerow({
                    "instance": filename, "formulation": "MTZ",
                    "obj_int": res_int['obj'], "time_int": res_int['time'],
                    "obj_relax": res_rel['obj'], "time_relax": res_rel['time'], # Ici on met les résultats de la relaxation
                    "gap": round(gap, 4), "vars": res_int['vars'], "constr": res_int['constr']
                })

            # --- 2. DFJ Enum (Entier + Relaxation) ---
            if n <= MAX_N_ENUM:
                res_int = run_solver(path, 2)
                res_rel = run_solver(path, 3)
                
                if res_int and res_rel:
                    gap = (res_int['obj'] - res_rel['obj']) / res_int['obj'] if res_int['obj'] else 0
                    
                    print(f"{filename:<25} {'DFJ_enum':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.2f} {gap:<8.4f} | {res_int['tour']}")
                    
                    writer.writerow({
                        "instance": filename, "formulation": "DFJ_enum",
                        "obj_int": res_int['obj'], "time_int": res_int['time'],
                        "obj_relax": res_rel['obj'], "time_relax": res_rel['time'],
                        "gap": round(gap, 4), "vars": res_int['vars'], "constr": res_int['constr']
                    })

            # --- 3. DFJ Iter (Standard uniquement) ---
            res_int = run_solver(path, 4)
            if res_int:
                print(f"{filename:<25} {'DFJ_iter':<10} {res_int['obj']:<10.2f} {res_int['time']:<8.2f} {'-':<8} | {res_int['tour']}")
                
                writer.writerow({
                    "instance": filename, "formulation": "DFJ_iter",
                    "obj_int": res_int['obj'], "time_int": res_int['time'],
                    "obj_relax": "", "time_relax": "",
                    "gap": "", "vars": res_int['vars'], "constr": res_int['constr']
                })
    
    print("-" * 100)
    print(f"Terminé ! Résultats sauvegardés dans {OUTPUT_CSV}")

if __name__ == "__main__":
    main()