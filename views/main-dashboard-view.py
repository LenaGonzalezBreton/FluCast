# views/main-dashboard-view.py
# ─────────────────────────────────────────────────────────────────────────────
# Vue unifiée : France / Grand Est — Prophet only + onglets Département & Région
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
sys.path.append(str(Path(__file__).resolve().parents[1]))  # racine du repo
try:
    from models import app as app_regional
    from models import app_national
except Exception:
    app_regional = None
    app_national = None

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Thermomètre Grippal Prédictif",
    page_icon="🌡️",
    layout="wide"
)

NATIONAL_CSV = Path("data/clean-data/donnees_analytiques_france.csv")
REGIONAL_CSV = Path("data/clean-data/donnees_analytiques_grand_est.csv")

# -----------------------------
# HELPERS : schéma & conversions
# -----------------------------
FALLBACK_CANDIDATES = {
    "code_departement": [
        "code_departement", "dep_code", "code_dep", "dep", "departement",
        "departement_code", "code", "num_dep", "code_insee", "insee_dep"
    ],
    "nom_departement": [
        "nom_departement", "nom", "departement_nom", "libelle_departement",
        "libelle", "dep_nom"
    ],
    "annee_semaine": [
        "annee_semaine", "semaine", "week", "year_week", "anneeSemaine",
        "yearweek", "YW", "yyyyww", "periode_semaine", "periode"
    ],
    "total_cas_semaine": [
        "total_cas_semaine", "total_cas", "cas_total", "cas", "nb_cas",
        "nombre_cas", "total", "sum_cas"
    ],
}

def _first_present(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Remap direct vers le schéma
    mapping = {}
    for target, candidates in FALLBACK_CANDIDATES.items():
        src = _first_present(df, candidates)
        if src and src != target:
            mapping[src] = target
    if mapping:
        df = df.rename(columns=mapping)

    # Si 'total_cas_semaine' absent, essayer de le créer à partir de colonnes proches
    if "total_cas_semaine" not in df.columns:
        urg_candidates = ["urgences", "nb_urgences", "passages_urgences", "urg", "er_visits",
                          "cas_urgences_semaine"]
        sos_candidates = ["sos_medecins", "nb_sos_medecins", "sos", "acts_sos", "sos_actes",
                          "cas_sos_medecins_semaine"]
        u = _first_present(df, urg_candidates)
        s = _first_present(df, sos_candidates)
        if u and s:
            df["total_cas_semaine"] = df[u].fillna(0).astype(float) + df[s].fillna(0).astype(float)
        elif u and "total_cas_semaine" not in df.columns:
            df["total_cas_semaine"] = df[u].fillna(0).astype(float)
        elif s and "total_cas_semaine" not in df.columns:
            df["total_cas_semaine"] = df[s].fillna(0).astype(float)

    # Normalisation des identifiants
    if "code_departement" in df.columns:
        df["code_departement"] = df["code_departement"].astype(str).str.upper().str.strip()
        mask_corse = df["code_departement"].str.contains("A|B")
        df.loc[~mask_corse, "code_departement"] = df.loc[~mask_corse, "code_departement"].str.zfill(2)
        df["code_departement"] = df["code_departement"].str.replace("0A", "2A").str.replace("0B", "2B")

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

def ensure_required_columns(df: pd.DataFrame) -> list[str]:
    required = ["code_departement", "nom_departement", "annee_semaine", "total_cas_semaine"]
    return [c for c in required if c not in df.columns]

# -----------------------------
# CHARGEMENT générique
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path)
    return coerce_schema(df)

def load_geojson_from_models() -> Optional[dict]:
    # Essaye les helpers des modules existants
    for mod in (app_national, app_regional):
        if mod:
            try:
                gj = mod.get_geojson()
                if gj:
                    return gj
            except Exception:
                pass
    # Fallback local si présent
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
    """
    Charge le CSV, entraîne Prophet par département, calcule le score, renvoie :
      - df_hist : historique complet (schéma coercé)
      - df_disp : dernière semaine par département avec colonnes prédictives
      - last_week_label : libellé de la dernière semaine utilisée
    """
    df_hist = load_csv(Path(csv_path_str))
    if df_hist.empty:
        return df_hist, pd.DataFrame(), None

    try:
        if vue_label.startswith("🇫🇷"):
            if not app_national:
                raise RuntimeError("Module app_national indisponible")
            df_loaded = app_national.charger_donnees(csv_path_str)  # gère zfill + Corse
            if df_loaded.empty:
                raise RuntimeError("Chargement national vide")
            df_pred = app_national.entrainer_et_predire(df_loaded)
            df_disp = app_national.calculer_score(df_pred)
        else:
            if not app_regional:
                raise RuntimeError("Module app (régional) indisponible")
            df_loaded = app_regional.charger_donnees(csv_path_str)
            if df_loaded.empty:
                raise RuntimeError("Chargement régional vide")
            df_pred = app_regional.entrainer_et_predire(df_loaded)
            df_disp = app_regional.calculer_score(df_pred)

        last_week_label = str(df_disp["annee_semaine"].iloc[0])
        return df_hist, df_disp, last_week_label

    except Exception as e:
        st.error(
            "Le moteur **Prophet** a rencontré une erreur et ne peut pas être remplacé par un fallback "
            "car le MVP a été retiré à votre demande.\n\n"
            f"**Détail**: {e}\n\n"
            "➡️ Sous **Windows**, si l’erreur mentionne `tbb.dll`, lancez `fix_tbb.bat` en administrateur puis relancez l’application."
        )
        return df_hist, pd.DataFrame(), None

df_full, df_display, last_week_label = run_prophet(vue, csv_path)

st.title("🌡️ Thermomètre Grippal Prédictif — Vue unifiée (Prophet)")

if df_full.empty or df_display.empty:
    st.stop()

missing = ensure_required_columns(df_full)
if missing:
    st.warning(f"Colonnes manquantes dans le CSV : {', '.join(missing)}")

geojson = load_geojson_from_models()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Carte & KPIs", "🏥 Analyse département", "🗺️ Analyse région", "ℹ️ À propos du projet"])

# --- TAB 1 : CARTE & KPIs ---
with tab1:
    label_vue = "France métropolitaine" if vue.startswith("🇫🇷") else "Région Grand Est"
    st.subheader(f"Carte de la tension — {label_vue} — semaine à venir (S+1)")
    if last_week_label:
        st.caption(f"Dernière semaine historique prise en compte : **{last_week_label}** · Moteur : **PROPHET**")

    # KPIs globaux
    k1, k2, k3 = st.columns(3)
    k1.metric("Score moyen", f"{df_display['score_global_predictif'].mean():.2f}")
    k2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(df_display['cas_predits_semaine_suivante']).sum())}")
    k3.metric("Tendance moyenne", f"{np.nanmean(df_display['tendance_evolution_cas']):+.1%}")

    # Carte
    if geojson is None:
        st.warning("Fond de carte GeoJSON introuvable. Fournissez `data/geo/departements.geojson` (clé `feature.properties.code`).")
    else:
        if vue.startswith("🇫🇷"):
            zoom = 4.6; center = {"lat": 46.6, "lon": 2.4}
        else:
            zoom = 6.5; center = {"lat": 48.6921, "lon": 6.1844}

        map_df = df_display.dropna(subset=["score_global_predictif"]).copy()

        fig = px.choropleth_mapbox(
            map_df,
            geojson=geojson,
            locations="code_departement",
            featureidkey="properties.code",
            color="score_global_predictif",
            color_continuous_scale=["#D6F5D6", "#FFF3B0", "#FFA500", "#FF5E5E", "#9E0031"],
            range_color=(0, 1),
            mapbox_style="carto-positron",
            zoom=zoom,
            center=center,
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
            **Couleur = score de tension prédictif** (0 → 1). Plus c'est **rouge**, plus le risque est élevé pour S+1.  
            L’info-bulle montre **cas prédits S+1**, **tendance S/S-1** et **score** exact.  
            Le moteur utilisé est **Prophet** (saisonnalités). 
            """
        )

# --- TAB 2 : ANALYSE DÉPARTEMENT ---
with tab2:
    st.subheader("Analyse détaillée par département")
    dep_options = sorted(df_display["nom_departement"].dropna().unique().tolist())
    dep_name = st.selectbox("Choisir un département", dep_options)

    dep_code = df_display.loc[df_display["nom_departement"] == dep_name, "code_departement"].iloc[0]

    df_hist_dep = df_full[df_full["code_departement"] == dep_code].copy()
    df_hist_dep["_week_date"] = df_hist_dep["annee_semaine"].apply(week_to_datetime)
    df_hist_dep = df_hist_dep.sort_values("_week_date")

    # KPIs département
    row = df_display[df_display["code_departement"] == dep_code].iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Score de tension", f"{row.get('score_global_predictif', np.nan):.2f}")
    k2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(row.get('cas_predits_semaine_suivante', 0)))}")
    k3.metric("Tendance des cas", f"{row.get('tendance_evolution_cas', 0.0):+.1%}")

    st.markdown("**Historique hebdomadaire des cas (urgences + SOS Médecins)**")
    if not df_hist_dep["_week_date"].isna().all():
        fig_ts = px.line(
            df_hist_dep,
            x="_week_date",
            y="total_cas_semaine",
            markers=True,
            labels={"_week_date": "Semaine", "total_cas_semaine": "Total cas/semaine"},
            title=f"Évolution des cas — {dep_name}",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("Impossible de tracer l'historique (format 'annee_semaine' non reconnu).")

    st.download_button(
        label="⬇️ Exporter les données du département (CSV)",
        data=df_hist_dep.drop(columns=["_week_date"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name=f"grippe_{dep_code}.csv",
        mime="text/csv",
    )

# --- TAB 3 : ANALYSE RÉGION ---
with tab3:
    st.subheader("Analyse régionale")
    # On ne force pas un mapping en dur : on exploite les colonnes si elles existent.
    region_col = "nom_region" if "nom_region" in df_full.columns else ("code_region" if "code_region" in df_full.columns else None)

    if region_col is None:
        st.warning(
            "Aucune colonne région détectée. Pour activer cette vue, ajoutez `nom_region` ou `code_region` "
            "dans le CSV analytique (ex. lors de la préparation des données)."
        )
    else:
        regions = sorted(df_full[region_col].dropna().unique().tolist())
        if not regions:
            st.info("Aucune région détectée dans les données.")
        else:
            default_region = "Grand Est" if "Grand Est" in regions else regions[0]
            selected_region = st.selectbox("Sélectionner une région", options=regions, index=regions.index(default_region))

            # Historique agrégé région (somme des cas)
            reg_hist = df_full[df_full[region_col] == selected_region].copy()
            reg_hist["_week_date"] = reg_hist["annee_semaine"].apply(week_to_datetime)
            reg_hist = reg_hist.dropna(subset=["_week_date"]).sort_values("_week_date")
            agg = (
                reg_hist.groupby("_week_date", as_index=False)["total_cas_semaine"]
                .sum()
                .rename(columns={"total_cas_semaine": "total_cas_region"})
            )

            # KPIs région actuels (depuis df_display, somme/synthèse)
            reg_disp = df_display.merge(
                df_full[[region_col, "code_departement"]].drop_duplicates(),
                on="code_departement", how="left"
            )
            reg_disp = reg_disp[reg_disp[region_col] == selected_region].copy()

            c1, c2, c3 = st.columns(3)
            c1.metric("Départements couverts", f"{reg_disp['code_departement'].nunique()}")
            c2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(reg_disp['cas_predits_semaine_suivante']).sum())}")
            c3.metric("Score moyen", f"{reg_disp['score_global_predictif'].mean():.2f}")

            # Courbe régionale
            st.markdown("**Historique agrégé de la région (cas/semaine)**")
            if not agg.empty:
                fig_r = px.line(
                    agg,
                    x="_week_date",
                    y="total_cas_region",
                    markers=True,
                    labels={"_week_date": "Semaine", "total_cas_region": "Cas région (somme)"},
                    title=f"Évolution des cas — {selected_region}",
                )
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.info("Pas assez de données pour tracer l'historique régional.")

            # Tableau des départements de la région, tri par score
            st.markdown("**Départements (triés par score décroissant)**")
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

# --- TAB 4 : À PROPOS DU PROJET ---
with tab4:
    st.header("À Propos du Projet")
    st.markdown("""
Ce projet a été réalisé dans le cadre du **Hackathon Santé Datalab x EPITECH**.

### Objectif
Développer un outil d’aide à la décision pour optimiser la stratégie vaccinale contre la grippe en **anticipant les zones de tension** sur le système de santé.

### Méthodologie (Prophet)
1. **Préparation des données** : consolidation hebdomadaire par département (urgences + SOS Médecins), standardisation des identifiants (codes INSEE), et enrichissements (couverture vaccinale, démographie…).
2. **Modélisation prédictive** : **Prophet** par département (saisonnalités annuelle/hebdomadaire si pertinent). La dernière prédiction **S+1** est consolidée pour l’affichage.
3. **Scoring de tension** : combinaison pondérée
   - **Cas prédits (S+1)** → normalisés
   - **Vulnérabilité vaccinale** → `1 - couverture`
   Le score final est borné en [0, 1].

### Utilisation
- **Vue** : *France métropolitaine* ou *Grand Est*
- **Carte** : score par département (plus rouge = plus de tension attendue).
- **Analyse Département** : historique local + KPIs.
- **Analyse Région** : KPIs, courbe agrégée, et classement des départements (si `nom_region`/`code_region` est présent dans les données).

### Notes techniques
- Si vous êtes sous **Windows** et rencontrez une erreur liée à **`tbb.dll`**, utilisez `fix_tbb.bat` (exécution **administrateur**), puis relancez l’app.
- CSV attendu (min) : `code_departement`, `nom_departement`, `annee_semaine`, `total_cas_semaine`.  
  Colonnes utilisées si présentes : `cas_predits_semaine_suivante`, `tendance_evolution_cas`, `score_global_predictif`, `nom_region`/`code_region`.

---
    """)
