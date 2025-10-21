# views/main-dashboard-view.py
# ─────────────────────────────────────────────────────────────────────────────
# Vue unifiée : France / Grand Est — Prophet only + #49C81B accent color
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Accès aux modules de modèles Prophet
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from models import app as app_regional
    from models import app_national
except Exception:
    app_regional = None
    app_national = None

# -----------------------------
# CONFIG GLOBALE
# -----------------------------
ACCENT_COLOR = "#49C81B"

st.set_page_config(
    page_title="Thermomètre Grippal Prédictif",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer un thème Streamlit via CSS
st.markdown(
    f"""
    <style>
    :root {{
        --accent: {ACCENT_COLOR};
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
        border-bottom: 3px solid var(--accent);
        color: var(--accent);
    }}
    div[data-testid="stMetricValue"] {{
        color: var(--accent);
    }}
    .stButton>button {{
        background-color: var(--accent);
        color: white !important;
        border: none;
        border-radius: 6px;
    }}
    .stButton>button:hover {{
        background-color: #3aaa17;
        transition: background-color 0.2s ease-in-out;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

NATIONAL_CSV = Path("data/clean-data/donnees_analytiques_france.csv")
REGIONAL_CSV = Path("data/clean-data/donnees_analytiques_grand_est.csv")

# -----------------------------
# HELPERS
# -----------------------------
def _first_present(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {}
    fallback = {
        "code_departement": ["dep", "code_dep", "departement_code"],
        "nom_departement": ["nom", "departement_nom"],
        "annee_semaine": ["semaine", "week", "year_week"],
        "total_cas_semaine": ["total_cas", "nb_cas"]
    }
    for target, candidates in fallback.items():
        src = _first_present(df, candidates)
        if src and src != target:
            mapping[src] = target
    if mapping:
        df = df.rename(columns=mapping)

    if "code_departement" in df.columns:
        df["code_departement"] = (
            df["code_departement"].astype(str).str.upper().str.strip().str.zfill(2)
        )
        df["code_departement"] = df["code_departement"].replace({"0A": "2A", "0B": "2B"})
    if "annee_semaine" in df.columns:
        df["annee_semaine"] = df["annee_semaine"].astype(str)
    return df

def week_to_datetime(week_str: str) -> Optional[pd.Timestamp]:
    try:
        s = str(week_str)
        if "-S" in s:
            year, sw = s.split("-S")
            return pd.to_datetime(f"{int(year)}-W{int(sw):02d}-1")
        if "W" in s:
            year, sw = s.split("W")
            return pd.to_datetime(f"{int(year)}-W{int(sw):02d}-1")
        if len(s) >= 6 and s[:4].isdigit():
            year, sw = int(s[:4]), int(s[4:])
            return pd.to_datetime(f"{year}-W{sw:02d}-1")
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return coerce_schema(df)

def load_geojson() -> Optional[dict]:
    for mod in (app_national, app_regional):
        if mod:
            try:
                gj = mod.get_geojson()
                if gj:
                    return gj
            except Exception:
                pass
    geo_path = Path("data/geo/departements.geojson")
    if geo_path.exists():
        with open(geo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Paramètres")

vue = st.sidebar.radio(
    "Vue",
    options=("🇫🇷 France métropolitaine", "🟣 Région Grand Est"),
    index=0,
    help="Bascule entre la vue nationale et la vue Grand Est."
)

st.sidebar.markdown("---")
default_csv = str(NATIONAL_CSV) if vue.startswith("🇫🇷") else str(REGIONAL_CSV)
csv_path = st.sidebar.text_input("Chemin du CSV analytique", value=default_csv)
st.sidebar.caption("Par défaut : data/clean-data/donnees_analytiques_*.csv")

# -----------------------------
# PIPELINE PROPHET
# -----------------------------
def run_prophet(vue_label: str, csv_path_str: str):
    df_hist = load_csv(Path(csv_path_str))
    if df_hist.empty:
        return df_hist, pd.DataFrame(), None

    if vue_label.startswith("🇫🇷"):
        df_loaded = app_national.charger_donnees(csv_path_str)
        df_pred = app_national.entrainer_et_predire(df_loaded)
        df_disp = app_national.calculer_score(df_pred)
    else:
        df_loaded = app_regional.charger_donnees(csv_path_str)
        df_pred = app_regional.entrainer_et_predire(df_loaded)
        df_disp = app_regional.calculer_score(df_pred)

    last_week_label = str(df_disp["annee_semaine"].iloc[0])
    return df_hist, df_disp, last_week_label

df_full, df_display, last_week_label = run_prophet(vue, csv_path)
geojson = load_geojson()

if df_full.empty or df_display.empty:
    st.error("Aucune donnée exploitable.")
    st.stop()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Carte & KPIs", "🏥 Analyse département", "🗺️ Analyse région", "ℹ️ À propos du projet"]
)

# --- TAB 1 ---
with tab1:
    label_vue = "France métropolitaine" if vue.startswith("🇫🇷") else "Région Grand Est"
    st.subheader(f"Carte de la tension — {label_vue} — semaine à venir (S+1)")
    if last_week_label:
        st.caption(f"Dernière semaine : **{last_week_label}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score moyen", f"{df_display['score_global_predictif'].mean():.2f}")
    c2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(df_display['cas_predits_semaine_suivante']).sum())}")
    c3.metric("Tendance moyenne", f"{np.nanmean(df_display['tendance_evolution_cas']):+.1%}")

    if geojson is not None:
        map_df = df_display.dropna(subset=["score_global_predictif"]).copy()
        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="code_departement",
            featureidkey="properties.code",
            color="score_global_predictif",
            color_continuous_scale=[
                "#E6F9E6", "#A8F0A8", "#49C81B", "#FFB84D", "#E63946"
            ],
            range_color=(0, 1),
            mapbox_style="carto-positron",
            zoom=4.8 if vue.startswith("🇫🇷") else 6.5,
            center={"lat": 46.6, "lon": 2.4} if vue.startswith("🇫🇷") else {"lat": 48.6, "lon": 6.1},
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
    else:
        st.warning("GeoJSON manquant.")

# --- TAB 2 ---
with tab2:
    st.subheader("Analyse détaillée par département")
    dep_options = sorted(df_display["nom_departement"].dropna().unique().tolist())
    dep_name = st.selectbox("Département", dep_options)

    dep_code = df_display.loc[df_display["nom_departement"] == dep_name, "code_departement"].iloc[0]
    df_dep = df_full[df_full["code_departement"] == dep_code].copy()
    df_dep["_week_date"] = df_dep["annee_semaine"].apply(week_to_datetime)
    df_dep = df_dep.sort_values("_week_date")

    row = df_display[df_display["code_departement"] == dep_code].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Score", f"{row.get('score_global_predictif', np.nan):.2f}")
    k2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(row.get('cas_predits_semaine_suivante', 0)))}")
    k3.metric("Tendance", f"{row.get('tendance_evolution_cas', 0.0):+.1%}")

    st.markdown("**Historique hebdomadaire des cas (urgences + SOS Médecins)**")
    fig_ts = px.line(
        df_dep,
        x="_week_date",
        y="total_cas_semaine",
        markers=True,
        labels={"_week_date": "Semaine", "total_cas_semaine": "Cas/semaine"},
        title=f"Évolution des cas — {dep_name}",
        line_shape="spline"
    )
    fig_ts.update_traces(line_color=ACCENT_COLOR)
    st.plotly_chart(fig_ts, use_container_width=True)

# --- TAB 3 ---
with tab3:
    st.subheader("Analyse régionale")
    region_col = "nom_region" if "nom_region" in df_full.columns else ("code_region" if "code_region" in df_full.columns else None)
    if not region_col:
        st.info("Aucune colonne région trouvée dans le CSV.")
    else:
        regions = sorted(df_full[region_col].dropna().unique().tolist())
        reg_sel = st.selectbox("Région", regions)
        reg_hist = df_full[df_full[region_col] == reg_sel].copy()
        reg_hist["_week_date"] = reg_hist["annee_semaine"].apply(week_to_datetime)
        agg = reg_hist.groupby("_week_date", as_index=False)["total_cas_semaine"].sum()
        st.metric("Cas prédits (S+1)",
                  f"{int(np.nan_to_num(df_display.merge(df_full[[region_col,'code_departement']].drop_duplicates(),on='code_departement')[lambda d: d[region_col]==reg_sel]['cas_predits_semaine_suivante']).sum())}")
        fig_r = px.line(
            agg,
            x="_week_date", y="total_cas_semaine",
            title=f"Évolution agrégée — {reg_sel}",
            markers=True
        )
        fig_r.update_traces(line_color=ACCENT_COLOR)
        st.plotly_chart(fig_r, use_container_width=True)

# --- TAB 4 ---
with tab4:
    st.header("À propos du projet")
    st.markdown(f"""
Ce projet a été réalisé dans le cadre du **Hackathon Santé Datalab x EPITECH**.

### 🎯 Objectif
Développer un outil d’aide à la décision pour **anticiper les zones de tension** grippales
et **optimiser la distribution des vaccins** et ressources médicales.

### 🧠 Méthodologie
- **Prophet** pour la prévision des cas hebdomadaires (saisonnalité annuelle).
- **Score de tension** = combinaison pondérée entre **cas prédits (S+1)** et **vulnérabilité vaccinale**.
- Visualisation interactive en **Streamlit + Plotly**, code Python pur.

### 🗺️ Fonctionnalités
- Carte de tension nationale ou régionale (code couleur basé sur {ACCENT_COLOR} → rouge).
- Analyses détaillées par **département** et **région**.
- Export CSV pour un partage rapide des données.

### ⚙️ Technique
- Python 3.11, Prophet, Pandas, Plotly, Streamlit.
- Exécution : `streamlit run views/main-dashboard-view.py`
- Couleur d’accent : **{ACCENT_COLOR}**
    """)
