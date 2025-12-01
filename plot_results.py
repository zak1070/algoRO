import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Lecture du CSV
df = pd.read_csv('results.csv')

# Filtrer pour n <= 15 (comme demandé pour l'analyse des relaxations)
# On extrait la taille n depuis le nom de l'instance
df['n'] = df['instance'].apply(lambda x: int(x.split('_')[1]))
df_small = df[df['n'] <= 15]

# On garde uniquement MTZ et DFJ_enum
df_analysis = df_small[df_small['formulation'].isin(['MTZ', 'DFJ_enum'])].copy()

# Nettoyage des gaps (parfois -0.0)
df_analysis['gap'] = df_analysis['gap'].abs()

# Configuration du style
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Création du barplot
ax = sns.barplot(
    data=df_analysis,
    x='instance',
    y='gap',
    hue='formulation',
    palette=['#3498db', '#e74c3c'] # Bleu et Rouge
)

# Customisation
plt.xticks(rotation=45, ha='right')
plt.title('Comparaison de l\'Integrality Gap : MTZ vs DFJ (n <= 15)', fontsize=14)
plt.ylabel('Gap (0.0 = Borne parfaite)', fontsize=12)
plt.xlabel('Instance', fontsize=12)
plt.legend(title='Formulation')
plt.tight_layout()

# Sauvegarde
plt.savefig('gap_analysis.png', dpi=300)
print("Graphique 'gap_analysis.png' généré avec succès !")
plt.show()