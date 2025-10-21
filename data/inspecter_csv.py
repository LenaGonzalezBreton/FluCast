# Fichier: inspecter_csv.py

import pandas as pd

# --- Mettez ici le chemin vers le fichier que vous voulez inspecter ---
# EXEMPLE 1:
chemin_du_fichier = "raw-data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"# EXEMPLE 2:
# chemin_du_fichier = "raw-data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"


try:
    # On essaie de lire le fichier avec le séparateur point-virgule
    df = pd.read_csv(chemin_du_fichier, sep=';', nrows=5)
    print(f"\n--- Inspection du fichier : {chemin_du_fichier} ---")
    print("\nNoms des colonnes trouvés :")
    print(df.columns.tolist())
    print("\nAperçu des 5 premières lignes :")
    print(df.head())
    print("\n---------------------------------------------------\n")

except FileNotFoundError:
    print(f"ERREUR: Le fichier '{chemin_du_fichier}' est introuvable. Vérifiez le nom et le chemin.")
except Exception as e:
    print(f"Une erreur est survenue: {e}")