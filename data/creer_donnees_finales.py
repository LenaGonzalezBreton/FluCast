# Fichier: creer_donnees_finales.py

import pandas as pd
import os
import numpy as np

# --- PARTIE 1: DÉFINITIONS ET CONFIGURATION ---

# Crée le dossier de sortie s'il n'existe pas
if not os.path.exists('clean-data'):
    os.makedirs('clean-data')

# Chemins vers les fichiers
CHEMIN_COUVERTURE_VACC = "raw-data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv"
CHEMIN_URGENCES = "raw-data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"
CHEMIN_SORTIE_FINAL = "clean-data/donnees_analytiques_grand_est.csv"

# Départements cibles
CODES_GRAND_EST = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']
DEPARTEMENTS_GRAND_EST = {
    '08': 'Ardennes', '10': 'Aube', '51': 'Marne', '52': 'Haute-Marne',
    '54': 'Meurthe-et-Moselle', '55': 'Meuse', '57': 'Moselle',
    '67': 'Bas-Rhin', '68': 'Haut-Rhin', '88': 'Vosges'
}

# --- PARTIE 2: FONCTIONS DE TRAITEMENT ---

def preparer_donnees_vaccination(chemin_fichier):
    print("-> Traitement des données de vaccination...")
    try:
        df = pd.read_csv(chemin_fichier, sep=',')
        df = df.rename(columns={
            'Département Code': 'code_departement',
            'Grippe 65 ans et plus': 'couv_vacc_grippe_an_passe'
        })
        df['code_departement'] = df['code_departement'].astype(str)
        df_ge = df[df['code_departement'].isin(CODES_GRAND_EST)].copy()
        df_ge['couv_vacc_grippe_an_passe'] = pd.to_numeric(df_ge['couv_vacc_grippe_an_passe'], errors='coerce') / 100
        df_final = df_ge[['code_departement', 'couv_vacc_grippe_an_passe']].drop_duplicates(subset='code_departement', keep='last')
        print("   ...Données de vaccination OK.")
        return df_final
    except (FileNotFoundError, KeyError) as e:
        print(f"   !!! ERREUR lors du traitement de la vaccination: {e}")
        return pd.DataFrame(columns=['code_departement', 'couv_vacc_grippe_an_passe'])

def preparer_donnees_urgences(chemin_fichier):
    print("-> Traitement des données des urgences et SOS Médecins...")
    try:
        # Le séparateur est bien une virgule
        df = pd.read_csv(chemin_fichier, sep=',')
        
        # On renomme la colonne département
        df = df.rename(columns={'Département Code': 'code_departement'})
        df['code_departement'] = df['code_departement'].astype(str)
        
        # On filtre sur les départements du Grand Est et la classe d'âge "Tous âges"
        df_ge = df[(df['code_departement'].isin(CODES_GRAND_EST)) & (df["Classe d'âge"] == 'Tous âges')].copy()
        
        # On renomme les colonnes de taux et on convertit en numérique
        df_ge = df_ge.rename(columns={
            'Semaine': 'annee_semaine',
            "Taux de passages aux urgences pour grippe": 'cas_urgences_semaine', # C'est un taux, mais on le traite comme un indicateur de cas
            "Taux d'actes médicaux SOS médecins pour grippe": 'cas_sos_medecins_semaine' # Idem
        })
        
        cols_to_numeric = ['cas_urgences_semaine', 'cas_sos_medecins_semaine']
        for col in cols_to_numeric:
            df_ge[col] = pd.to_numeric(df_ge[col], errors='coerce').fillna(0)

        df_ge['total_cas_semaine'] = df_ge['cas_urgences_semaine'] + df_ge['cas_sos_medecins_semaine']
        
        # Calcul de la tendance
        df_ge = df_ge.sort_values(['code_departement', 'annee_semaine'])
        df_ge['total_cas_semaine_precedente'] = df_ge.groupby('code_departement')['total_cas_semaine'].shift(1)
        df_ge['tendance_evolution_cas'] = ((df_ge['total_cas_semaine'] - df_ge['total_cas_semaine_precedente']) / df_ge['total_cas_semaine_precedente']).fillna(0)
        df_ge.replace([np.inf, -np.inf], 0, inplace=True)

        print("   ...Données des urgences OK.")
        return df_ge[['code_departement', 'annee_semaine', 'cas_urgences_semaine', 'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas']]
    except (FileNotFoundError, KeyError) as e:
        print(f"   !!! ERREUR lors du traitement des urgences: {e}")
        return pd.DataFrame()

def enrichir_donnees_insee(df):
    print("-> Enrichissement avec les données INSEE (simulation)...")
    pop_data = {
        '08': {'pop_totale': 270000, 'densite': 52}, '10': {'pop_totale': 310000, 'densite': 51},
        '51': {'pop_totale': 565000, 'densite': 70}, '52': {'pop_totale': 172000, 'densite': 28},
        '54': {'pop_totale': 733000, 'densite': 139}, '55': {'pop_totale': 184000, 'densite': 30},
        '57': {'pop_totale': 1045000, 'densite': 167},'67': {'pop_totale': 1140000, 'densite': 237},
        '68': {'pop_totale': 767000, 'densite': 215}, '88': {'pop_totale': 364000, 'densite': 62}
    }
    df['population_totale'] = df['code_departement'].map(lambda x: pop_data.get(x, {}).get('pop_totale', 0))
    df['densite_population'] = df['code_departement'].map(lambda x: pop_data.get(x, {}).get('densite', 0))
    df['population_plus_65_ans'] = (df['population_totale'] * 0.21).astype(int)
    df['pct_plus_65_ans'] = 0.21
    print("   ...Données INSEE simulées OK.")
    return df

def ajouter_predictions_modele(df):
    print("-> Ajout des prédictions du modèle (simulation)...")
    df['cas_predits_semaine_suivante'] = (df['total_cas_semaine'] * (1 + df['tendance_evolution_cas'] * 0.5)).astype(int)
    print("   ...Prédictions simulées OK.")
    return df

def calculer_score_final(df):
    print("-> Calcul du score de tension final (simulation)...")
    df['score_cas_norm'] = df['cas_predits_semaine_suivante'] / df['cas_predits_semaine_suivante'].max() if df['cas_predits_semaine_suivante'].max() > 0 else 0
    df['score_vacc_norm'] = 1 - df['couv_vacc_grippe_an_passe']
    df['score_global_predictif'] = np.clip(0.6 * df['score_cas_norm'] + 0.4 * df['score_vacc_norm'], 0, 1)
    df['score_global_predictif'] = df['score_global_predictif'].round(2)
    print("   ...Score final calculé OK.")
    return df

# --- PARTIE 3: SCRIPT PRINCIPAL ---

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("Lancement de la préparation du dataset HISTORIQUE...")
    
    donnees_vacc = preparer_donnees_vaccination(CHEMIN_COUVERTURE_VACC)
    donnees_urg = preparer_donnees_urgences(CHEMIN_URGENCES)

    if not donnees_urg.empty:
        df_final = pd.merge(donnees_urg, donnees_vacc, on='code_departement', how='left')
        df_final['couv_vacc_grippe_an_passe'] = df_final.groupby('code_departement')['couv_vacc_grippe_an_passe'].ffill().bfill()
        df_final = enrichir_donnees_insee(df_final)

        df_final['nom_departement'] = df_final['code_departement'].map(DEPARTEMENTS_GRAND_EST)
        
        # On garde uniquement les colonnes de l'historique
        colonnes_historiques = [
            'code_departement', 'nom_departement', 'annee_semaine', 'population_totale',
            'population_plus_65_ans', 'pct_plus_65_ans', 'densite_population',
            'couv_vacc_grippe_an_passe', 'cas_urgences_semaine',
            'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas'
        ]
        df_final = df_final[colonnes_historiques]

        df_final.to_csv(CHEMIN_SORTIE_FINAL, index=False, sep=';')
        print(f"\n✅ Le fichier historique '{CHEMIN_SORTIE_FINAL}' est prêt !")