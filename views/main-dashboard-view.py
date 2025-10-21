import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Thermomètre Grippal Prédictif",
    page_icon="🌡️",
    layout="wide"
)

DATA_PATH = Path("data/clean-data/donnees_analytiques_france.csv")
GEOJSON_PATH = Path("data/geo/departements.geojson")

# -----------------------------
# HELPERS & CACHE
# -----------------------------
# === Schéma flexible : détection/renommage automatique ===
FALLBACK_CANDIDATES = {
    "code_departement": [
        "code_departement", "dep_code", "code_dep", "dep", "departement", "departement_code",
        "code", "num_dep", "code_insee", "insee_dep"
    ],
    "nom_departement": [
        "nom_departement", "nom", "departement_nom", "libelle_departement", "libelle", "dep_nom"
    ],
    "annee_semaine": [
        "annee_semaine", "semaine", "week", "year_week", "anneeSemaine", "yearweek", "YW", "yyyyww",
        "periode_semaine", "periode"
    ],
    "total_cas_semaine": [
        "total_cas_semaine", "total_cas", "cas_total", "cas", "nb_cas", "nombre_cas",
        "total", "sum_cas"
    ],
}

def _first_present(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Essaie d'inférer les 4 colonnes requises et les (re)crée si possible.
    - Remappe les noms de colonnes "proches" vers le schéma attendu
    - Si total_cas_semaine est absent, tente = urgences + sos_medecins (ou colonnes approchantes)
    - Normalise code dep (zfill, 2A/2B)
    - Force annee_semaine en str
    """
    df = df.copy()

    # Mapping direct des 4 colonnes requises
    mapping = {}
    for target, candidates in FALLBACK_CANDIDATES.items():
        src = _first_present(df, candidates)
        if src and src != target:
            mapping[src] = target
    if mapping:
        df = df.rename(columns=mapping)

    # Si 'total_cas_semaine' absent, essayer de le constituer
    if "total_cas_semaine" not in df.columns:
        # heuristiques sur urgences + SOS Médecins
        urg_candidates = [
            "urgences", "nb_urgences", "passages_urgences", "urg", "er_visits"
        ]
        sos_candidates = [
            "sos_medecins", "nb_sos_medecins", "sos", "acts_sos", "sos_actes"
        ]
        u = _first_present(df, urg_candidates)
        s = _first_present(df, sos_candidates)
        if u and s:
            df["total_cas_semaine"] = df[u].fillna(0).astype(float) + df[s].fillna(0).astype(float)
        elif u and "total_cas_semaine" not in df.columns:
            df["total_cas_semaine"] = df[u].fillna(0).astype(float)
        elif s and "total_cas_semaine" not in df.columns:
            df["total_cas_semaine"] = df[s].fillna(0).astype(float)

    # Normalisation types/champs
    if "code_departement" in df.columns:
        df["code_departement"] = (
            df["code_departement"].astype(str).str.upper().str.strip()
        )
        # Gérer formats type '1' -> '01'; conserver '2A/2B'
        df.loc[~df["code_departement"].str.contains("A|B"), "code_departement"] = (
            df.loc[~df["code_departement"].str.contains("A|B"), "code_departement"].str.zfill(2)
        )
        df["code_departement"] = df["code_departement"].str.replace("0A", "2A").str.replace("0B", "2B")

    if "annee_semaine" in df.columns:
        df["annee_semaine"] = df["annee_semaine"].astype(str)

    return df

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = coerce_schema(df)
    return df

@st.cache_data(show_spinner=False)
def load_geojson(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def week_to_datetime(week_str: str) -> Optional[pd.Timestamp]:
    """Convertit 'YYYY-SWW' en date (lundi de la semaine)."""
    try:
        # Supporte '2025-S04', '2025W04', '202504' etc.
        s = str(week_str)
        if "S" in s:
            year, sw = s.split("-S")
            return pd.to_datetime(f"{year}-W{int(sw):02d}-1")
        elif "W" in s:
            year, sw = s.split("W")
            return pd.to_datetime(f"{year}-W{int(sw):02d}-1")
        else:
            # ex: 202542 -> 2025 semaine 42
            if len(s) >= 6:
                year, sw = int(s[:4]), int(s[4:])
                return pd.to_datetime(f"{year}-W{sw:02d}-1")
    except Exception:
        return None
    return None

def ensure_required_columns(df: pd.DataFrame) -> list[str]:
    required = ["code_departement", "nom_departement", "annee_semaine", "total_cas_semaine"]
    missing = [c for c in required if c not in df.columns]
    return missing

def compute_latest_week_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Trie par semaine décroissante (en utilisant conversion date pour robustesse)
    df = df.copy()
    df["_week_date"] = df["annee_semaine"].apply(week_to_datetime)
    df = df.sort_values(["_week_date"], ascending=False)
    latest_date = df["_week_date"].dropna().max()
    if pd.isna(latest_date):
        # fallback: premier élément après tri lexicographique
        latest_week = df["annee_semaine"].iloc[0]
        return df[df["annee_semaine"] == latest_week]
    latest_week = df.loc[df["_week_date"].idxmax(), "annee_semaine"]
    return df[df["annee_semaine"] == latest_week]

def minmax_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng

def compute_light_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si les colonnes prédictives n'existent pas, on calcule un MVP :
    - Prédiction S+1 = moyenne glissante des 3 dernières semaines
    - Tendance = (S0 - S-1) / S-1
    - Score = min-max des prédictions (échelle [0,1])
    """
    df = df.copy()

    # Prépare une table agrégée par (dep, semaine)
    base = df[["code_departement", "nom_departement", "annee_semaine", "total_cas_semaine"]].copy()
    base["_week_date"] = base["annee_semaine"].apply(week_to_datetime)
    base = base.dropna(subset=["_week_date"]).sort_values(["code_departement", "_week_date"])

    # Rolling 3 semaines pour prédiction S+1
    base["_roll3"] = base.groupby("code_departement")["total_cas_semaine"].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # Tendance S/S-1
    base["_lag1"] = base.groupby("code_departement")["total_cas_semaine"].shift(1)
    base["tendance_evolution_cas"] = (base["total_cas_semaine"] - base["_lag1"]) / base["_lag1"]
    base.loc[~np.isfinite(base["tendance_evolution_cas"]) | base["tendance_evolution_cas"].isna(), "tendance_evolution_cas"] = 0.0

    # Dernière semaine dispo par dep
    latest = base.sort_values(["code_departement", "_week_date"]).groupby("code_departement").tail(1)
    latest = latest.rename(columns={"_roll3": "cas_predits_semaine_suivante"})

    # Score via min-max sur les prédictions
    latest["score_global_predictif"] = minmax_series(latest["cas_predits_semaine_suivante"])

    # On mappe ces valeurs sur le df d'origine (pour garder compat UI en aval)
    pred = latest[[
        "code_departement", "nom_departement", "annee_semaine",
        "cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"
    ]]

    return pred

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Paramètres")

csv_path = st.sidebar.text_input("Chemin du CSV analytique", value=str(DATA_PATH))
geojson_path = st.sidebar.text_input("Chemin du GeoJSON départements", value=str(GEOJSON_PATH))

# Filtrage optionnel par région si présent dans les données
st.sidebar.markdown("—")
show_regions = st.sidebar.checkbox("Activer le filtre par région (si disponible)", value=True)

# -----------------------------
# CHARGEMENT
# -----------------------------
df = load_csv(Path(csv_path))
geojson = load_geojson(Path(geojson_path))

st.title("🌡️ Thermomètre Grippal Prédictif — France métropolitaine")

if df.empty:
    st.error("Aucune donnée chargée. Vérifie le chemin du CSV.")
    st.stop()

missing = ensure_required_columns(df)
if missing:
    st.error(f"Colonnes manquantes dans le CSV : {', '.join(missing)}")
    st.stop()

# Filtre région (si colonnes dispo)
if show_regions and ("nom_region" in df.columns or "code_region" in df.columns):
    region_col = "nom_region" if "nom_region" in df.columns else "code_region"
    regions = sorted(df[region_col].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect("Filtrer par région", options=regions, default=[])
    if selected_regions:
        df = df[df[region_col].isin(selected_regions)]

# Si les prédictions/score sont absents, les calculer en léger
need_pred = any(c not in df.columns for c in ["cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"])
if need_pred:
    pred = compute_light_predictions(df)
else:
    # Utiliser la dernière semaine dispo telle quelle
    pred = df[[
        "code_departement", "nom_departement", "annee_semaine",
        "cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"
    ]].dropna().copy()

# Vue "dernière semaine" pour la carte
df_latest = compute_latest_week_frame(df)
# Joindre les indicateurs prédit/score (qui eux sont sur la dernière semaine par dep)
df_latest = df_latest.merge(
    pred[["code_departement", "cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"]],
    on="code_departement", how="left"
)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ Vue Carte", "🏥 Analyse département", "ℹ️ À propos"])

# --- TAB 1 : CARTE ---
with tab1:
    st.subheader("Carte de la tension — semaine à venir (S+1)")
    if geojson is None:
        st.warning("GeoJSON introuvable. Fournis 'data/geo/departements.geojson' (feature.properties.code)")
    else:
        # KPIs globaux
        col1, col2, col3 = st.columns(3)
        col1.metric("Score moyen national", f"{df_latest['score_global_predictif'].mean():.2f}")
        col2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(df_latest['cas_predits_semaine_suivante']).sum())}")
        col3.metric("Tendance moyenne", f"{np.nanmean(df_latest['tendance_evolution_cas']):+.1%}")

        map_df = df_latest.dropna(subset=["score_global_predictif"]).copy()
        # Affichage carte
        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="code_departement",
            featureidkey="properties.code",
            color="score_global_predictif",
            color_continuous_scale=["#D6F5D6", "#FFF3B0", "#FFA500", "#FF5E5E", "#9E0031"],
            range_color=(0, 1),
            mapbox_style="carto-positron",
            zoom=4.6,
            center={"lat": 46.6, "lon": 2.4},
            opacity=0.75,
            hover_name="nom_departement",
            hover_data={
                "code_departement": False,
                "cas_predits_semaine_suivante": True,
                "tendance_evolution_cas": ":.1%",
                "score_global_predictif": ":.2f",
            },
        )
        fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Comment lire la carte ?"):
            st.markdown(
                """
                **Couleur = score de tension prédictif** (0 → 1). Plus c'est **rouge**, plus le risque est élevé la semaine prochaine.
                Les info-bulles affichent : **cas prédits S+1**, **tendance** S/S-1, et **score** exact.
                """
            )

# --- TAB 2 : ANALYSE DEPARTEMENT ---
with tab2:
    st.subheader("Analyse détaillée par département")
    dep_options = sorted(df_latest["nom_departement"].dropna().unique().tolist())
    dep_name = st.selectbox("Choisir un département", dep_options)

    dep_code = df_latest.loc[df_latest["nom_departement"] == dep_name, "code_departement"].iloc[0]

    # Série historique pour ce département
    dep_hist = df[df["code_departement"] == dep_code].copy()
    dep_hist["_week_date"] = dep_hist["annee_semaine"].apply(week_to_datetime)
    dep_hist = dep_hist.sort_values("_week_date")

    # Indicateurs actuels
    row = df_latest[df_latest["code_departement"] == dep_code].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Score de tension", f"{row.get('score_global_predictif', np.nan):.2f}")
    k2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(row.get('cas_predits_semaine_suivante', 0)))}")
    k3.metric("Tendance des cas", f"{row.get('tendance_evolution_cas', 0.0):+.1%}")

    # Série temporelle
    st.markdown("**Historique hebdomadaire des cas (urgences + SOS Médecins)**")
    if not dep_hist["_week_date"].isna().all():
        fig_ts = px.line(
            dep_hist,
            x="_week_date",
            y="total_cas_semaine",
            markers=True,
            labels={"_week_date": "Semaine", "total_cas_semaine": "Total cas/semaine"},
            title=f"Évolution des cas — {dep_name}",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Impossible de tracer l'historique (format 'annee_semaine' non reconnu).")

    # Export des données filtrées
    st.download_button(
        label="⬇️ Exporter les données du département (CSV)",
        data=dep_hist.drop(columns=["_week_date"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name=f"grippe_{dep_code}.csv",
        mime="text/csv",
    )

# --- TAB 3 : A PROPOS ---
with tab3:
    st.markdown(
        """
        ### Objectif
        Anticiper les **zones de tension** liées à la grippe et **aider à l'allocation** des vaccins et ressources.

        ### Méthodo (MVP)
        - Utilisation d'un **CSV analytique** consolidé (France entière, par département, hebdomadaire).
        - Si les prédictions/score sont absents, calcul léger : **moyenne glissante 3 semaines** pour S+1, 
          **tendance** S/S-1 et **score** min-max en [0,1].
        - Carte choroplèthe (Plotly) + analyses départementales.

        ### Données & extensions
        - Ajout possible de régresseurs (météo, pollution, Google Trends, logistique) et d'un modèle **Prophet** en backend
          pour des prédictions robustes et un **Score de Tension Vaccinale** plus riche.
        """
    )

    with st.expander("Checklist d'installation"):
        st.markdown(
            """
            1. Créer et activer un environnement Python 3.10+  
               `python -m venv .venv && source .venv/bin/activate` (Windows: `./.venv/Scripts/activate`)
            2. Installer les dépendances minimales  
               `pip install streamlit pandas plotly`
            3. Placer les fichiers :  
               - CSV : `data/clean-data/donnees_analytiques_france.csv`  
               - GeoJSON : `data/geo/departements.geojson`
            4. Lancer  
               `streamlit run app.py`
            """
        )

    with st.expander("Format du CSV attendu"):
        st.code(
            """code_departement,nom_departement,annee_semaine,total_cas_semaine,cas_predits_semaine_suivante,tendance_evolution_cas,score_global_predictif,nom_region\n"
            "88,Vosges,2025-S41,123,140,0.12,0.73,Grand Est\n"
            "54,Meurthe-et-Moselle,2025-S41,200,210,0.05,0.54,Grand Est\n",
            language="csv"
        """
        )
