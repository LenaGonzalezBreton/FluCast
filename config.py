"""
Configuration file for FluCast application.
Contains all configurable parameters for data processing, modeling, and visualization.
"""

import os
from pathlib import Path

# === PATHS ===
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DATA_DIR = DATA_DIR / "clean-data"
RAW_DATA_DIR = DATA_DIR / "raw-data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"

# Data file paths
DONNEES_FRANCE_PATH = CLEAN_DATA_DIR / "donnees_analytiques_france.csv"
DONNEES_GRAND_EST_PATH = CLEAN_DATA_DIR / "donnees_analytiques_grand_est.csv"

# === GEOGRAPHIC PARAMETERS ===
# Metropolitan French departments (codes)
CODES_METRO = [str(d).zfill(2) for d in range(1, 96)]
CODES_METRO.remove('20')  # Remove old Corsica numbering
CODES_METRO.extend(['2A', '2B'])  # Add new Corsica codes

# Grand Est region departments
CODES_GRAND_EST = ['08', '10', '51', '52', '54', '55', '57', '67', '68', '88']

# Department names for Grand Est
DEPARTEMENTS_GRAND_EST = {
    '08': 'Ardennes', 
    '10': 'Aube', 
    '51': 'Marne', 
    '52': 'Haute-Marne',
    '54': 'Meurthe-et-Moselle', 
    '55': 'Meuse', 
    '57': 'Moselle',
    '67': 'Bas-Rhin', 
    '68': 'Haut-Rhin', 
    '88': 'Vosges'
}

# === PROPHET MODEL PARAMETERS ===
PROPHET_CONFIG = {
    'seasonality_mode': 'additive',
    'weekly_seasonality': False,
    'daily_seasonality': False,
    'yearly_seasonality': True,
    'changepoint_prior_scale': 0.05,  # Controls flexibility of trend
    'seasonality_prior_scale': 10.0,   # Controls flexibility of seasonality
    'interval_width': 0.95             # Uncertainty interval width
}

# Number of weeks to predict
PREDICTION_PERIODS = 1
PREDICTION_FREQ = 'W'

# === SCORING PARAMETERS ===
# Weights for global tension score calculation
SCORE_WEIGHTS = {
    'cas_predits': 0.6,      # Weight for predicted cases
    'vulnerabilite_vacc': 0.4 # Weight for vaccination vulnerability
}

# Score normalization bounds
SCORE_MIN = 0.0
SCORE_MAX = 1.0

# === VISUALIZATION PARAMETERS ===
# Map configuration
MAP_CONFIG = {
    'mapbox_style': 'carto-positron',
    'center': {"lat": 46.8566, "lon": 2.3522},  # Center of France
    'zoom': 4.5,
    'opacity': 0.7,
    'color_scale': 'YlOrRd',  # Yellow-Orange-Red color scale
    'color_range': (0, 1)
}

# GeoJSON URL for French departments
GEOJSON_URL = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"

# === DATA PROCESSING PARAMETERS ===
# CSV separator for clean data files
CSV_SEPARATOR = ';'

# Columns expected in the final dataset
EXPECTED_COLUMNS = [
    'code_departement',
    'nom_departement',
    'annee_semaine',
    'population_totale',
    'population_plus_65_ans',
    'pct_plus_65_ans',
    'densite_population',
    'couv_vacc_grippe_an_passe',
    'cas_urgences_semaine',
    'cas_sos_medecins_semaine',
    'total_cas_semaine',
    'tendance_evolution_cas'
]

# === LOGGING PARAMETERS ===
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# === CACHE PARAMETERS ===
# Enable Streamlit caching
ENABLE_CACHE = True

# Cache TTL (Time To Live) in seconds
CACHE_TTL = 3600  # 1 hour

# === VALIDATION PARAMETERS ===
# Minimum number of data points required for Prophet model
MIN_DATA_POINTS = 3

# Maximum acceptable percentage of missing values
MAX_MISSING_PCT = 0.3  # 30%

# Valid range for vaccination coverage (0-1)
VACC_COVERAGE_RANGE = (0.0, 1.0)

# Valid range for case numbers
CASE_NUMBER_RANGE = (0, float('inf'))
