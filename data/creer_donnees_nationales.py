# Fichier: creer_donnees_nationales.py
# Ce script prépare les données pour la France entière.

import pandas as pd
import os
import numpy as np

# --- CONFIGURATION ---
if not os.path.exists('clean-data'):
    os.makedirs('clean-data')

CHEMIN_COUVERTURE_VACC = "raw-data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv"
CHEMIN_URGENCES = "raw-data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"
CHEMIN_SORTIE_FINAL = "clean-data/donnees_analytiques_france.csv" # Nouveau nom de fichier

# --- FONCTIONS DE TRAITEMENT (identiques à avant) ---

def preparer_donnees_vaccination(chemin_fichier):
    print("-> Traitement des données de vaccination (France entière)...")
    try:
        df = pd.read_csv(chemin_fichier, sep=',')
        df = df.rename(columns={
            'Département Code': 'code_departement',
            'Grippe 65 ans et plus': 'couv_vacc_grippe_an_passe'
        })
        df['code_departement'] = df['code_departement'].astype(str)
        
        # On ne filtre PAS sur le Grand Est pour garder tous les départements
        
        df['couv_vacc_grippe_an_passe'] = pd.to_numeric(df['couv_vacc_grippe_an_passe'], errors='coerce') / 100
        df_final = df[['code_departement', 'couv_vacc_grippe_an_passe']].drop_duplicates(subset='code_departement', keep='last')
        print("   ...Données de vaccination OK.")
        return df_final
    except (FileNotFoundError, KeyError) as e:
        print(f"   !!! ERREUR vaccination: {e}")
        return pd.DataFrame(columns=['code_departement', 'couv_vacc_grippe_an_passe'])

def preparer_donnees_urgences(chemin_fichier):
    print("-> Traitement des données des urgences (France entière)...")
    try:
        df = pd.read_csv(chemin_fichier, sep=',')
        df = df.rename(columns={'Département Code': 'code_departement'})
        df['code_departement'] = df['code_departement'].astype(str)
        
        # On ne filtre PAS sur le Grand Est
        df_ge = df[df["Classe d'âge"] == 'Tous âges'].copy()
        
        df_ge = df_ge.rename(columns={
            'Semaine': 'annee_semaine',
            "Taux de passages aux urgences pour grippe": 'cas_urgences_semaine',
            "Taux d'actes médicaux SOS médecins pour grippe": 'cas_sos_medecins_semaine'
        })
        
        cols_to_numeric = ['cas_urgences_semaine', 'cas_sos_medecins_semaine']
        for col in cols_to_numeric:
            df_ge[col] = pd.to_numeric(df_ge[col], errors='coerce').fillna(0)

        df_ge['total_cas_semaine'] = df_ge['cas_urgences_semaine'] + df_ge['cas_sos_medecins_semaine']
        
        df_ge = df_ge.sort_values(['code_departement', 'annee_semaine'])
        df_ge['total_cas_semaine_precedente'] = df_ge.groupby('code_departement')['total_cas_semaine'].shift(1)
        df_ge['tendance_evolution_cas'] = ((df_ge['total_cas_semaine'] - df_ge['total_cas_semaine_precedente']) / df_ge['total_cas_semaine_precedente']).fillna(0)
        df_ge.replace([np.inf, -np.inf], 0, inplace=True)

        print("   ...Données des urgences OK.")
        return df_ge[['code_departement', 'annee_semaine', 'cas_urgences_semaine', 'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas']]
    except (FileNotFoundError, KeyError) as e:
        print(f"   !!! ERREUR urgences: {e}")
        return pd.DataFrame()

def enrichir_donnees_insee(df, df_urgences_source):
    print("-> Enrichissement avec les données INSEE et noms (simulation)...")
    # Fusionner pour obtenir les noms des départements
    noms_deps = df_urgences_source[['code_departement', 'Département']].drop_duplicates()
    df = pd.merge(df, noms_deps, on='code_departement', how='left')
    df = df.rename(columns={'Département': 'nom_departement'})

    # ## TODO ## ÉQUIPE DATA : Remplacez cette section par le chargement
    # et la fusion de votre VRAI fichier INSEE national.
    df['population_totale'] = np.random.randint(100000, 1200000, size=len(df))
    df['densite_population'] = np.random.randint(30, 500, size=len(df))
    df['population_plus_65_ans'] = (df['population_totale'] * 0.21).astype(int)
    df['pct_plus_65_ans'] = 0.21
    print("   ...Données INSEE simulées OK.")
    return df

# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    print("Lancement de la création du dataset analytique pour la FRANCE ENTIÈRE...")
    
    # On a besoin du fichier urgences aussi pour les noms de département
    df_urgences_source = pd.read_csv(CHEMIN_URGENCES, sep=',', usecols=['Département Code', 'Département']).rename(columns={'Département Code': 'code_departement'})
    df_urgences_source['code_departement'] = df_urgences_source['code_departement'].astype(str)

    donnees_vacc = preparer_donnees_vaccination(CHEMIN_COUVERTURE_VACC)
    donnees_urg = preparer_donnees_urgences(CHEMIN_URGENCES)

    if not donnees_urg.empty:
        df_final = pd.merge(donnees_urg, donnees_vacc, on='code_departement', how='left')
        df_final['couv_vacc_grippe_an_passe'] = df_final.groupby('code_departement')['couv_vacc_grippe_an_passe'].ffill().bfill()

        df_final = enrichir_donnees_insee(df_final, df_urgences_source)

        colonnes_finales = [
            'code_departement', 'nom_departement', 'annee_semaine', 'population_totale',
            'population_plus_65_ans', 'pct_plus_65_ans', 'densite_population',
            'couv_vacc_grippe_an_passe', 'cas_urgences_semaine',
            'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas'
        ]
        df_final = df_final[colonnes_finales]

        df_final.to_csv(CHEMIN_SORTIE_FINAL, index=False, sep=';')

        print(f"\n✅ Le fichier national '{CHEMIN_SORTIE_FINAL}' a été créé avec succès !")
    else:
        print("\n !!! Arrêt du script: Données d'urgences non chargées.")
