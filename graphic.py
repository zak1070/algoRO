import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_file):
    df = pd.read_csv(csv_file)
    
    # Nettoyage pour les graphiques
    df = df[df['time_int'] != ''] # Filtrer les erreurs
    df['n'] = df['instance'].apply(lambda x: int(x.split('_')[1]))
    df = df.sort_values('n')

    # 1. Temps de résolution (Log Scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='time_int', hue='formulation', marker='o')
    plt.yscale('log')
    plt.title('Comparaison des temps de résolution (Log Scale)')
    plt.ylabel('Temps (secondes)')
    plt.xlabel('Nombre de villes (n)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('time_comparison.png')
    plt.show()

    # 2. Nombre de contraintes (Log Scale)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='constr', hue='formulation', marker='o')
    plt.yscale('log')
    plt.title('Explosion du nombre de contraintes')
    plt.ylabel('Nombre de contraintes')
    plt.xlabel('Nombre de villes (n)')
    plt.grid(True, which="both", ls="--")
    plt.savefig('constraints_comparison.png')
    plt.show()

if __name__ == "__main__":
    # Assurez-vous d'avoir généré results.csv avec benchmark.py avant
    try:
        plot_results("results.csv")
    except FileNotFoundError:
        print("Fichier results.csv manquant.")