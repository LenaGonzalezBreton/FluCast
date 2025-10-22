# Fichier: creer_donnees_nationales.py
# Ce script prépare les données pour la France entière.

import pandas as pd
import os
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import utils
import config

# Set up logger
logger = utils.setup_logger(__name__, 'data_preparation.log')

# --- CONFIGURATION ---
LOGS_DIR = Path('data/logs')
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_DATA_DIR = Path('clean-data')
CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

CHEMIN_COUVERTURE_VACC = "raw-data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv"
CHEMIN_URGENCES = "raw-data/grippe-passages-aux-urgences-et-actes-sos-medecins-departement.csv"
CHEMIN_SORTIE_FINAL = CLEAN_DATA_DIR / "donnees_analytiques_france.csv"

# --- FONCTIONS DE TRAITEMENT (identiques à avant) ---

def preparer_donnees_vaccination(chemin_fichier):
    """
    Prepare vaccination coverage data from CSV file.
    
    Args:
        chemin_fichier: Path to vaccination data CSV file
    
    Returns:
        DataFrame with vaccination data by department
    """
    logger.info("Processing vaccination data (France)...")
    
    try:
        df = pd.read_csv(chemin_fichier, sep=',')
        df = df.rename(columns={
            'Département Code': 'code_departement',
            'Grippe 65 ans et plus': 'couv_vacc_grippe_an_passe'
        })
        df['code_departement'] = df['code_departement'].astype(str)
        
        # Convert vaccination coverage to decimal (0-1)
        df['couv_vacc_grippe_an_passe'] = utils.safe_numeric_conversion(
            df['couv_vacc_grippe_an_passe']
        ) / 100
        
        # Keep only latest value per department
        df_final = df[['code_departement', 'couv_vacc_grippe_an_passe']].drop_duplicates(
            subset='code_departement', 
            keep='last'
        )
        
        # Validate vaccination coverage range
        is_valid, error = utils.validate_numeric_range(
            df_final['couv_vacc_grippe_an_passe'],
            *config.VACC_COVERAGE_RANGE,
            'couv_vacc_grippe_an_passe'
        )
        
        if not is_valid:
            logger.warning(error)
        
        logger.info(f"Vaccination data OK: {len(df_final)} departments")
        return df_final
        
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error processing vaccination data: {e}")
        return pd.DataFrame(columns=['code_departement', 'couv_vacc_grippe_an_passe'])

def preparer_donnees_urgences(chemin_fichier):
    """
    Prepare emergency and SOS Médecins data from CSV file.
    
    Args:
        chemin_fichier: Path to urgences data CSV file
    
    Returns:
        DataFrame with emergency data by department and week
    """
    logger.info("Processing emergency data (France)...")
    
    try:
        df = pd.read_csv(chemin_fichier, sep=',')
        df = df.rename(columns={'Département Code': 'code_departement'})
        df['code_departement'] = df['code_departement'].astype(str)
        
        # Filter for all ages
        df_ge = df[df["Classe d'âge"] == 'Tous âges'].copy()
        
        df_ge = df_ge.rename(columns={
            'Semaine': 'annee_semaine',
            "Taux de passages aux urgences pour grippe": 'cas_urgences_semaine',
            "Taux d'actes médicaux SOS médecins pour grippe": 'cas_sos_medecins_semaine'
        })
        
        # Safe numeric conversion
        df_ge['cas_urgences_semaine'] = utils.safe_numeric_conversion(
            df_ge['cas_urgences_semaine']
        )
        df_ge['cas_sos_medecins_semaine'] = utils.safe_numeric_conversion(
            df_ge['cas_sos_medecins_semaine']
        )
        
        df_ge['total_cas_semaine'] = (
            df_ge['cas_urgences_semaine'] + 
            df_ge['cas_sos_medecins_semaine']
        )
        
        # Calculate week-over-week trend
        df_ge = df_ge.sort_values(['code_departement', 'annee_semaine'])
        df_ge['total_cas_semaine_precedente'] = (
            df_ge.groupby('code_departement')['total_cas_semaine'].shift(1)
        )
        df_ge['tendance_evolution_cas'] = utils.calculate_percentage_change(
            df_ge['total_cas_semaine'],
            df_ge['total_cas_semaine_precedente']
        )

        logger.info(f"Emergency data OK: {len(df_ge)} records")
        return df_ge[[
            'code_departement', 'annee_semaine', 'cas_urgences_semaine', 
            'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas'
        ]]
        
    except (FileNotFoundError, KeyError) as e:
        logger.error(f"Error processing emergency data: {e}")
        return pd.DataFrame()

def enrichir_donnees_insee(df, df_urgences_source):
    """
    Enrich data with INSEE demographic information.
    
    Currently uses simulated data. 
    TODO: Replace with real INSEE data loading and merging.
    
    Args:
        df: DataFrame to enrich
        df_urgences_source: Source data with department names
    
    Returns:
        Enriched DataFrame with demographic columns
    """
    logger.info("Enriching with INSEE data (simulated)...")
    
    # Merge department names
    noms_deps = df_urgences_source[['code_departement', 'Département']].drop_duplicates()
    df = pd.merge(df, noms_deps, on='code_departement', how='left')
    df = df.rename(columns={'Département': 'nom_departement'})

    # ## TODO ## ÉQUIPE DATA : Remplacez cette section par le chargement
    # et la fusion de votre VRAI fichier INSEE national.
    logger.warning(
        "Using SIMULATED demographic data. "
        "Replace with real INSEE data for production use."
    )
    
    # Simulate demographic data
    np.random.seed(42)  # For reproducible simulations
    df['population_totale'] = np.random.randint(100000, 1200000, size=len(df))
    df['densite_population'] = np.random.randint(30, 500, size=len(df))
    df['population_plus_65_ans'] = (df['population_totale'] * 0.21).astype(int)
    df['pct_plus_65_ans'] = 0.21
    
    logger.info("INSEE data enrichment complete (simulated)")
    return df

# --- SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("Starting national dataset creation for FRANCE")
    logger.info("="*60)
    
    # Load source data for department names
    try:
        df_urgences_source = pd.read_csv(
            CHEMIN_URGENCES, 
            sep=',', 
            usecols=['Département Code', 'Département']
        ).rename(columns={'Département Code': 'code_departement'})
        df_urgences_source['code_departement'] = df_urgences_source['code_departement'].astype(str)
    except Exception as e:
        logger.error(f"Failed to load source data: {e}")
        sys.exit(1)

    # Process vaccination data
    donnees_vacc = preparer_donnees_vaccination(CHEMIN_COUVERTURE_VACC)
    
    # Process emergency data
    donnees_urg = preparer_donnees_urgences(CHEMIN_URGENCES)

    if donnees_urg.empty:
        logger.error("Emergency data is empty. Stopping.")
        sys.exit(1)
    
    # Merge vaccination and emergency data
    logger.info("Merging datasets...")
    df_final = pd.merge(donnees_urg, donnees_vacc, on='code_departement', how='left')
    
    # Forward/backward fill vaccination coverage per department
    df_final['couv_vacc_grippe_an_passe'] = (
        df_final.groupby('code_departement')['couv_vacc_grippe_an_passe']
        .ffill()
        .bfill()
    )

    # Enrich with INSEE data
    df_final = enrichir_donnees_insee(df_final, df_urgences_source)

    # Select final columns
    colonnes_finales = [
        'code_departement', 'nom_departement', 'annee_semaine', 'population_totale',
        'population_plus_65_ans', 'pct_plus_65_ans', 'densite_population',
        'couv_vacc_grippe_an_passe', 'cas_urgences_semaine',
        'cas_sos_medecins_semaine', 'total_cas_semaine', 'tendance_evolution_cas'
    ]
    df_final = df_final[colonnes_finales]

    # Save to CSV
    success = utils.safe_write_csv(
        df_final, 
        CHEMIN_SORTIE_FINAL, 
        separator=';',
        logger=logger
    )
    
    if success:
        logger.info("="*60)
        logger.info(f"✅ National dataset created successfully!")
        logger.info(f"   File: {CHEMIN_SORTIE_FINAL}")
        logger.info(f"   Rows: {len(df_final)}")
        logger.info(f"   Departments: {df_final['code_departement'].nunique()}")
        logger.info(f"   Weeks: {df_final['annee_semaine'].nunique()}")
        logger.info("="*60)
    else:
        logger.error("Failed to save output file")
        sys.exit(1)
