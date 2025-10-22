"""
National flu forecasting model module.
Provides functions for training Prophet models, making predictions,
and calculating tension scores for French departments.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

import streamlit as st
import pandas as pd
import requests
from prophet import Prophet
from typing import Optional, Dict

import config
import utils

# Set up logger
logger = utils.setup_logger(__name__, 'app_national.log')


# === MODEL TRAINING AND PREDICTION ===

@st.cache_data
def entrainer_et_predire(df_historique: pd.DataFrame) -> pd.DataFrame:
    """
    Train a Prophet model for each department and return predictions.
    Includes fallback handling for optimization errors.
    
    Args:
        df_historique: Historical data DataFrame with columns:
            - code_departement: Department code
            - annee_semaine: Week in format 'YYYY-SXX'
            - total_cas_semaine: Total cases for the week
    
    Returns:
        DataFrame with predictions for each department, containing:
            - All columns from the most recent week
            - cas_predits_semaine_suivante: Predicted cases for next week
    """
    logger.info("Starting model training and prediction...")
    
    predictions_futures = []

    # Keep only metropolitan departments for the map
    df_metro = df_historique[
        df_historique['code_departement'].isin(config.CODES_METRO)
    ].copy()
    
    logger.info(f"Processing {len(df_metro['code_departement'].unique())} departments")

    for code_dep in df_metro['code_departement'].unique():
        
        df_dep = df_metro[df_metro['code_departement'] == code_dep].copy()
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame()
        cleaned_week = df_dep['annee_semaine'].str.replace('S', '')
        df_prophet['ds'] = pd.to_datetime(
            cleaned_week + '-1', 
            format='%Y-%U-%w', 
            errors='coerce'
        )
        df_prophet['y'] = df_dep['total_cas_semaine'].values
        df_prophet = df_prophet.dropna(subset=['ds'])

        # Check if we have enough data points
        if len(df_prophet) < config.MIN_DATA_POINTS:
            logger.warning(
                f"Department {code_dep}: Insufficient data ({len(df_prophet)} points). "
                f"Minimum required: {config.MIN_DATA_POINTS}"
            )
            continue
        
        try:
            # Train Prophet model
            modele = Prophet(**config.PROPHET_CONFIG)
            
            # Suppress Prophet's verbose output
            with utils.suppress_stdout_stderr():
                modele.fit(df_prophet)

            # Make prediction
            future = modele.make_future_dataframe(
                periods=config.PREDICTION_PERIODS, 
                freq=config.PREDICTION_FREQ
            )
            forecast = modele.predict(future)
            
            # Extract prediction (ensure non-negative)
            prediction_valeur = max(0, int(forecast.iloc[-1]['yhat']))
            
            logger.debug(f"Department {code_dep}: Predicted {prediction_valeur} cases")

        except Exception as e:
            # Fallback: use simple prediction (last observed value)
            logger.warning(
                f"Department {code_dep}: Prophet failed ({type(e).__name__}). "
                f"Using simple fallback prediction."
            )
            prediction_valeur = int(df_prophet.iloc[-1]['y'])

        predictions_futures.append({
            'code_departement': code_dep,
            'cas_predits_semaine_suivante': prediction_valeur
        })

    if not predictions_futures:
        logger.error("No predictions could be generated")
        derniere_semaine = df_metro['annee_semaine'].max()
        return df_metro[df_metro['annee_semaine'] == derniere_semaine].copy()
    
    # Merge predictions with current week data
    df_predictions = pd.DataFrame(predictions_futures)
    derniere_semaine = df_metro['annee_semaine'].max()
    df_actuel = df_metro[df_metro['annee_semaine'] == derniere_semaine].copy()
    
    # Remove existing prediction column if present
    if 'cas_predits_semaine_suivante' in df_actuel.columns:
        df_actuel = df_actuel.drop(columns=['cas_predits_semaine_suivante'])
        
    df_final_predit = pd.merge(df_actuel, df_predictions, on='code_departement')
    
    logger.info(f"Predictions completed for {len(df_predictions)} departments")
    return df_final_predit

@st.cache_data
def calculer_score(df_predit: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the final tension score from enriched data.
    
    The tension score combines:
    - Predicted case numbers (normalized)
    - Vaccination vulnerability (1 - vaccination coverage)
    
    Args:
        df_predit: DataFrame with predictions, must contain:
            - cas_predits_semaine_suivante: Predicted cases
            - couv_vacc_grippe_an_passe: Vaccination coverage (0-1)
    
    Returns:
        DataFrame with added score columns:
            - score_cas_norm: Normalized predicted cases (0-1)
            - score_vacc_norm: Vaccination vulnerability (0-1)
            - score_global_predictif: Combined tension score (0-1)
    """
    logger.info("Calculating tension scores...")
    
    # Normalize predicted cases
    max_cas_predits = df_predit['cas_predits_semaine_suivante'].max()
    if max_cas_predits > 0:
        df_predit['score_cas_norm'] = (
            df_predit['cas_predits_semaine_suivante'] / max_cas_predits
        )
    else:
        logger.warning("No predicted cases found, setting normalized score to 0")
        df_predit['score_cas_norm'] = 0
        
    # Calculate vaccination vulnerability (inverse of coverage)
    df_predit['score_vacc_norm'] = 1 - df_predit['couv_vacc_grippe_an_passe']
    
    # Weighted combination using config weights
    w_cas = config.SCORE_WEIGHTS['cas_predits']
    w_vacc = config.SCORE_WEIGHTS['vulnerabilite_vacc']
    
    df_predit['score_global_predictif'] = (
        w_cas * df_predit['score_cas_norm'] + 
        w_vacc * df_predit['score_vacc_norm']
    ).clip(config.SCORE_MIN, config.SCORE_MAX)
    
    df_predit['score_global_predictif'] = (
        df_predit['score_global_predictif'].round(2)
    )
    
    logger.info(
        f"Scores calculated. Mean score: "
        f"{df_predit['score_global_predictif'].mean():.2f}"
    )
    
    return df_predit

@st.cache_resource
def get_geojson() -> Optional[Dict]:
    """
    Download and return GeoJSON for French departments.
    
    Returns:
        GeoJSON dictionary if successful, None otherwise
    """
    logger.info(f"Downloading GeoJSON from {config.GEOJSON_URL}")
    
    try:
        r = requests.get(config.GEOJSON_URL, timeout=10)
        r.raise_for_status()
        logger.info("GeoJSON downloaded successfully")
        return r.json()
    except requests.exceptions.Timeout:
        error_msg = "Timeout while downloading GeoJSON"
        logger.error(error_msg)
        st.error(f"{error_msg}. Please check your internet connection.")
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to download GeoJSON: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return None

@st.cache_data
def charger_donnees(chemin_fichier: str) -> pd.DataFrame:
    """
    Load the clean historical dataset.
    
    Args:
        chemin_fichier: Path to the CSV file
    
    Returns:
        DataFrame with loaded data, or empty DataFrame on error
    """
    logger.info(f"Loading data from {chemin_fichier}")
    
    try:
        df = pd.read_csv(chemin_fichier, sep=config.CSV_SEPARATOR)
        
        # Standardize department codes
        df['code_departement'] = (
            df['code_departement']
            .astype(str)
            .str.zfill(2)
        )
        
        # Handle Corsica for GeoJSON merge
        df['code_departement'] = df['code_departement'].replace({'20': '2A'})
        
        # Validate data
        is_valid, errors = utils.validate_dataframe(
            df, 
            config.EXPECTED_COLUMNS,
            name="Dataset"
        )
        
        if not is_valid:
            for error in errors:
                logger.error(error)
                st.error(error)
        
        # Check for missing data
        missing = utils.check_missing_data(df)
        if missing:
            logger.warning(f"Missing data detected: {missing}")
        
        logger.info(f"Loaded {len(df)} rows, {len(df['code_departement'].unique())} departments")
        return df
        
    except FileNotFoundError:
        error_msg = f"File not found: {chemin_fichier}"
        logger.error(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
    except Exception as e:
        error_msg = f"Error loading data: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return pd.DataFrame()
