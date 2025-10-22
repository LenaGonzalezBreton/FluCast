# views/main-dashboard-view.py
# ─────────────────────────────────────────────────────────────────────────────
# Vue unifiée : France / Grand Est — Prophet only
# Esthétique : accent #49C81B + logo (header) ; carte YlOrRd ; mode Clair/Sombre
# ─────────────────────────────────────────────────────────────────────────────

import json
from datetime import datetime, timedelta
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
    page_icon="data/assets/logo_icon_app.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CONSTANTES
# -----------------------------
NATIONAL_CSV = Path("data/clean-data/donnees_analytiques_france.csv")
REGIONAL_CSV = Path("data/clean-data/donnees_analytiques_grand_est.csv")
DEFAULT_LOGO = Path("data/assets/logo.png")  # logo par défaut (modifiable dans la sidebar)


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
        "code_departement": ["dep", "code_dep", "departement_code", "departement", "code"],
        "nom_departement": ["nom", "departement_nom", "libelle_departement"],
        "annee_semaine": ["semaine", "week", "year_week"],
        "total_cas_semaine": ["total_cas", "nb_cas", "cas_total"]
    }
    for target, candidates in fallback.items():
        src = _first_present(df, candidates)
        if src and src != target:
            mapping[src] = target
    if mapping:
        df = df.rename(columns=mapping)

    if "code_departement" in df.columns:
        df["code_departement"] = df["code_departement"].astype(str).str.upper().str.strip()
        mask_corse = df["code_departement"].str.contains("A|B")
        df.loc[~mask_corse, "code_departement"] = df.loc[~mask_corse, "code_departement"].str.zfill(2)
        df["code_departement"] = df["code_departement"].replace({"0A": "2A", "0B": "2B"})
    if "annee_semaine" in df.columns:
        df["annee_semaine"] = df["annee_semaine"].astype(str)
    return df

def week_to_datetime(week_str: str) -> Optional[pd.Timestamp]:
    try:
        s = str(week_str)
        year = None
        week = None

        # Format "2024-S01" ou "2024-S1"
        if "-S" in s:
            year, sw = s.split("-S")
            year, week = int(year), int(sw)
        # Format "2024W01" ou "2024W1"
        elif "W" in s:
            year, sw = s.split("W")
            year, week = int(year), int(sw)
        # Format "202401" (6+ chiffres)
        elif len(s) >= 6 and s[:4].isdigit():
            year, week = int(s[:4]), int(s[4:])

        if year and week:
            # Méthode ISO : le 4 janvier est toujours dans la semaine 1
            jan_4 = datetime(year, 1, 4)
            # Trouver le lundi de la semaine 1
            week_1_monday = jan_4 - timedelta(days=jan_4.weekday())
            # Ajouter les semaines nécessaires
            target_date = week_1_monday + timedelta(weeks=week - 1)
            return pd.Timestamp(target_date)

    except Exception as e:
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
st.sidebar.title("Paramètres")

# Thème clair / sombre (par défaut clair)
theme_choice = st.sidebar.radio("Thème", ("Clair", "Sombre"), index=1)
# Applique le thème (CSS + Plotly + Mapbox)
if theme_choice == "Clair":
    # CSS clair
    st.markdown(
        f"""
        <style>
        :root {{ --accent: {ACCENT_COLOR}; }}
        .stApp {{ background-color: #ffffff; color: #111827; }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 3px solid var(--accent);
            color: var(--accent);
        }}
        div[data-testid="stMetricValue"] {{ color: var(--accent); }}
        .stButton>button {{
            background-color: var(--accent); color: white !important;
            border: none; border-radius: 6px;
        }}
        .stButton>button:hover {{ background-color: #3aaa17; transition: background-color .2s; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = "plotly"          # clair
    MAPBOX_STYLE = "carto-positron"          # clair
else:
    st.markdown(
        f"""
        <style>
        :root {{ --accent: {ACCENT_COLOR}; }}
        .stApp {{ background-color: #0f172a; color: #f8fafc; }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            border-bottom: 3px solid var(--accent);
            color: var(--accent);
        }}
        div[data-testid="stMetricValue"] {{ color: var(--accent); }}
        .stButton>button {{
            background-color: var(--accent); color: white !important;
            border: none; border-radius: 6px;
        }}
        .stButton>button:hover {{ background-color: #3aaa17; transition: background-color .2s; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    px.defaults.template = "plotly_dark"     # sombre
    MAPBOX_STYLE = "carto-darkmatter"        # sombre

st.sidebar.markdown("---")

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

st.sidebar.markdown("---")
logo_path_str = st.sidebar.text_input("Logo (PNG/JPG)", value=str(DEFAULT_LOGO))

# 🔥 NOTE : aucune case “afficher aussi le logo dans la sidebar” — supprimée.

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

# -----------------------------
# ENTÊTE (logo + titre)
# -----------------------------
hdr_col_logo, hdr_col_title = st.columns([0.12, 0.88])
logo_path = Path("data/assets/logo.png")
with hdr_col_logo:
    if logo_path.exists():
        try:
            st.logo(str(logo_path))
        except Exception as e:
            st.write(f"Erreur st.logo: {e}")  # Pour voir l'erreur
            st.image(str(logo_path), use_container_width=True)

with hdr_col_title:
    st.title("Thermomètre Grippal Prédictif — Vue unifiée (Prophet)")

if df_full.empty or df_display.empty:
    st.error("Aucune donnée exploitable.")
    st.stop()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Carte & KPIs", "Analyse département", "Analyse région", "À propos du projet"]
)

# --- TAB 1 : Carte & KPIs ---
with tab1:
    label_vue = "France métropolitaine" if vue.startswith("🇫🇷") else "Région Grand Est"
    st.subheader(f"Carte de la tension — {label_vue} — semaine à venir (S+1)")
    if last_week_label:
        st.caption(f"Dernière semaine : **{last_week_label}** · Moteur : **PROPHET**")

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
            color_continuous_scale="YlOrRd",  # ✅ nuances d’orange conservées
            range_color=(0, 1),
            mapbox_style=MAPBOX_STYLE,        # clair/sombre selon le thème
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
        st.warning("GeoJSON manquant pour la carte.")

# --- TAB 2 : Analyse département ---
with tab2:
    st.subheader("Analyse détaillée par département")
    dep_options = sorted(df_display["nom_departement"].dropna().unique().tolist())
    dep_name = st.selectbox("Département", dep_options)

    dep_code = df_display.loc[df_display["nom_departement"] == dep_name, "code_departement"].iloc[0]
    df_dep = df_full[df_full["code_departement"] == dep_code].copy()
    df_dep["_week_date"] = df_dep["annee_semaine"].apply(week_to_datetime)
    df_dep = df_dep[df_dep["_week_date"].notna()].copy()
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

# --- TAB 3 : Analyse région ---
with tab3:
    st.subheader("Analyse régionale")
    region_col = "nom_region" if "nom_region" in df_full.columns else ("code_region" if "code_region" in df_full.columns else None)
    if not region_col:
        st.info("Aucune colonne région trouvée dans le CSV (ajoutez `nom_region` ou `code_region`).")
    else:
        regions = sorted(df_full[region_col].dropna().unique().tolist())
        reg_sel = st.selectbox("Région", regions)
        reg_hist = df_full[df_full[region_col] == reg_sel].copy()
        reg_hist["_week_date"] = reg_hist["annee_semaine"].apply(week_to_datetime)
        agg = reg_hist.groupby("_week_date", as_index=False)["total_cas_semaine"].sum()

        reg_disp = df_display.merge(
            df_full[[region_col, "code_departement"]].drop_duplicates(),
            on="code_departement", how="left"
        )
        reg_disp = reg_disp[reg_disp[region_col] == reg_sel].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Départements couverts", f"{reg_disp['code_departement'].nunique()}")
        c2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(reg_disp['cas_predits_semaine_suivante']).sum())}")
        c3.metric("Score moyen", f"{reg_disp['score_global_predictif'].mean():.2f}")

        st.markdown("**Historique agrégé de la région (cas/semaine)**")
        if not agg.empty:
            fig_r = px.line(
                agg, x="_week_date", y="total_cas_semaine",
                title=f"Évolution agrégée — {reg_sel}", markers=True
            )
            fig_r.update_traces(line_color=ACCENT_COLOR)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Pas assez de données pour tracer l'historique régional.")

        st.markdown("**Départements de la région (triés par score décroissant)**")
        table_cols = [
            "code_departement", "nom_departement",
            "cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"
        ]
        table = (
            reg_disp[table_cols]
            .sort_values("score_global_predictif", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(table, use_container_width=True)

# --- TAB 4 : À propos du projet ---
with tab4:
    st.header("À propos du projet")
    st.markdown(f"""
Ce projet a été réalisé dans le cadre du **Hackathon Santé Datalab x EPITECH**.

###  Objectif
Anticiper les **zones de tension grippale** et aider à la **répartition des vaccins** en temps réel.

###  Méthodologie
- Modélisation **Prophet** par département.
- Score global basé sur les cas prédits et la vulnérabilité vaccinale.
- Visualisation interactive via **Streamlit + Plotly**.

###  Identité visuelle
- Accent couleur : **{ACCENT_COLOR}**
- Carte : **nuances d’orange (YlOrRd)** cohérentes avec l’indicateur de tension.
- Logo en-tête (modifiable via la sidebar). Mode **{theme_choice.lower()}** actif.

###  Technique
- Python 3.11 · Prophet · Pandas · Plotly · Streamlit
- Lancement : `streamlit run views/main-dashboard-view.py`


###  L'équipe 
 - https://www.linkedin.com/in/axel-momper
 - https://www.linkedin.com/in/lucas-olivarez/
 - https://www.linkedin.com/in/alexy-pinto-3b4765301/
 - https://www.linkedin.com/in/maxence-noizet-2a4a2526b/
 - https://www.linkedin.com/in/lenagonzalezbreton/

    """)




