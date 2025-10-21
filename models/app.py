import sys

sys.path.append('../T-HAK-700-NCY_6/data/clean-data')

import streamlit as st
import pandas as pd
import requests # Pour télécharger le fond de carte
from prophet import Prophet # Importez Prophet au début


# --- FONCTIONS DU MODÈLE ET DU SCORING ---

@st.cache_data # Cache la sortie pour ne pas ré-entraîner à chaque interaction
def entrainer_et_predire(df_historique):
    """
    Entraîne un modèle Prophet pour chaque département et retourne les prédictions.
    Inclut un filet de sécurité pour gérer les erreurs d'optimisation.
    """
    print("-> Lancement de l'entraînement et de la prédiction réelle...")
    
    predictions_futures = []

    for code_dep in df_historique['code_departement'].unique():
        
        df_dep = df_historique[df_historique['code_departement'] == code_dep].copy()
        df_prophet = pd.DataFrame()
        
        cleaned_week = df_dep['annee_semaine'].str.replace('S', '')
        df_prophet['ds'] = pd.to_datetime(cleaned_week + '-1', format='%Y-%U-%w', errors='coerce')
        df_prophet['y'] = df_dep['total_cas_semaine'].values

        df_prophet = df_prophet.dropna(subset=['ds'])

        if len(df_prophet) > 2:
            try:
                # On essaie d'entraîner le modèle Prophet
                modele = Prophet(
                    seasonality_mode='additive',
                    weekly_seasonality=False, 
                    daily_seasonality=False, 
                    yearly_seasonality=True
                )
                modele.fit(df_prophet)

                future = modele.make_future_dataframe(periods=1, freq='W')
                forecast = modele.predict(future)
                
                prediction_valeur = max(0, int(forecast.iloc[-1]['yhat']))

            except RuntimeError as e:
                # Si le modèle plante, on utilise une prédiction simple comme filet de sécurité
                print(f"   !!! AVERTISSEMENT: Le modèle Prophet a échoué pour le département {code_dep}. Utilisation d'une prédiction simple.")
                prediction_valeur = int(df_prophet.iloc[-1]['y'])

            predictions_futures.append({
                'code_departement': code_dep,
                'cas_predits_semaine_suivante': prediction_valeur
            })

    if predictions_futures:
        df_predictions = pd.DataFrame(predictions_futures)
        
        derniere_semaine = df_historique['annee_semaine'].max()
        df_actuel = df_historique[df_historique['annee_semaine'] == derniere_semaine].copy()
        
        if 'cas_predits_semaine_suivante' in df_actuel.columns:
            df_actuel = df_actuel.drop(columns=['cas_predits_semaine_suivante'])
            
        df_final_predit = pd.merge(df_actuel, df_predictions, on='code_departement')
        
        print("   ...Prédictions réelles terminées.")
        return df_final_predit
    else:
        print("   ...Pas assez de données pour faire des prédictions.")
        return df_historique[df_historique['annee_semaine'] == df_historique['annee_semaine'].max()].copy()

@st.cache_data
def calculer_score(df_predit):
    """Calcule le score de tension final à partir des données enrichies."""
    print("-> Calcul du score de tension...")
    
    max_cas_predits = df_predit['cas_predits_semaine_suivante'].max()
    if max_cas_predits > 0:
        df_predit['score_cas_norm'] = df_predit['cas_predits_semaine_suivante'] / max_cas_predits
    else:
        df_predit['score_cas_norm'] = 0
        
    df_predit['score_vacc_norm'] = 1 - df_predit['couv_vacc_grippe_an_passe']
    
    # Combinaison pondérée : 60% du poids sur les cas prédits, 40% sur la vulnérabilité vaccinale
    df_predit['score_global_predictif'] = (0.6 * df_predit['score_cas_norm'] + 0.4 * df_predit['score_vacc_norm']).clip(0, 1)
    df_predit['score_global_predictif'] = df_predit['score_global_predictif'].round(2)
    return df_predit

@st.cache_resource
def get_geojson():
    """Télécharge et retourne le GeoJSON des départements français."""
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de télécharger le fond de carte : {e}")
        return None

@st.cache_data
def charger_donnees(chemin_fichier):
    """Charge le dataset historique propre."""
    try:
        df = pd.read_csv(chemin_fichier, sep=';')
        df['code_departement'] = df['code_departement'].astype(str).str.zfill(2)
        return df
    except FileNotFoundError:
        st.error(f"ERREUR : Le fichier '{chemin_fichier}' est introuvable.")
        return pd.DataFrame()