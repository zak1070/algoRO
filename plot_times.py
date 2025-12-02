import matplotlib
matplotlib.use("Agg")  # pas d'interface graphique
import pandas as pd
import matplotlib.pyplot as plt

# 1. Charger les résultats
df = pd.read_csv("results.csv")

# 2. Extraire n à partir du nom de l'instance, ex: instance_10_circle_1.txt
def extract_n(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except Exception:
        return None

df["n"] = df["instance"].apply(extract_n)

# 3. Ne garder que les trois formulations qui nous intéressent
df = df[df["formulation"].isin(["MTZ", "DFJ_enum", "DFJ_iter"])]

# 4. Agréger par (formulation, n) -> médiane du temps
agg = (
    df.groupby(["formulation", "n"], as_index=False)["time_int"]
      .median()
      .rename(columns={"time_int": "time_median"})
)

# 5. Tracer
plt.figure(figsize=(8, 4))

for form in ["DFJ_enum", "DFJ_iter", "MTZ"]:
    sub = agg[agg["formulation"] == form]
    plt.plot(sub["n"], sub["time_median"], marker="o", label=form)

plt.xlabel("Taille de l'instance $n$")
plt.ylabel("Temps de résolution (s)")
plt.title("Temps de résolution selon la taille et la formulation")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("times_plot.png", dpi=300)
# pas de plt.show() pour éviter Tkinter
