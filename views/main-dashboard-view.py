# views/main-dashboard-view.py
# ─────────────────────────────────────────────────────────────────────────────
# Vue unifiée : France / Grand Est + Moteur prédictif (MVP rapide / Prophet)
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Accès aux modules de modèles (Prophet)
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

# Chemins par défaut (conformes à ton repo)
NATIONAL_CSV = Path("data/clean-data/donnees_analytiques_france.csv")
REGIONAL_CSV = Path("data/clean-data/donnees_analytiques_grand_est.csv")

# GeoJSON : on le prend via les helpers existants des modules si possible.
def load_geojson_from_models() -> Optional[dict]:
    # Essaye national, puis régional
    for mod in (app_national, app_regional):
        if mod:
            try:
                gj = mod.get_geojson()
                if gj:
                    return gj
            except Exception:
                pass
    return None

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
    # Remap direct
    mapping = {}
    for target, candidates in FALLBACK_CANDIDATES.items():
        src = _first_present(df, candidates)
        if src and src != target:
            mapping[src] = target
    if mapping:
        df = df.rename(columns=mapping)

    # Si 'total_cas_semaine' absent, essayer de le créer
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

    # Normalisation types/champs
    if "code_departement" in df.columns:
        df["code_departement"] = df["code_departement"].astype(str).str.upper().str.strip()
        # Zero-fill sauf Corse
        mask_corse = df["code_departement"].str.contains("A|B")
        df.loc[~mask_corse, "code_departement"] = df.loc[~mask_corse, "code_departement"].str.zfill(2)
        # Normaliser 2A/2B
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

def minmax_series(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / rng

def compute_light_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    MVP fallback :
    - Prédiction S+1 = moyenne glissante des 3 dernières semaines
    - Tendance = (S0 - S-1) / S-1
    - Score = min-max des prédictions par rapport aux autres deps (échelle [0,1])
    Renvoie un DataFrame (1 ligne / département) pour la dernière semaine de chaque dep.
    """
    df = df.copy()
    base = df[["code_departement", "nom_departement", "annee_semaine", "total_cas_semaine"]].copy()
    base["_week_date"] = base["annee_semaine"].apply(week_to_datetime)
    base = base.dropna(subset=["_week_date"]).sort_values(["code_departement", "_week_date"])

    base["_roll3"] = base.groupby("code_departement")["total_cas_semaine"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    base["_lag1"] = base.groupby("code_departement")["total_cas_semaine"].shift(1)
    base["tendance_evolution_cas"] = (base["total_cas_semaine"] - base["_lag1"]) / base["_lag1"]
    base.loc[~np.isfinite(base["tendance_evolution_cas"]) | base["tendance_evolution_cas"].isna(),
             "tendance_evolution_cas"] = 0.0

    latest = base.groupby("code_departement", as_index=False).tail(1).copy()
    latest = latest.rename(columns={"_roll3": "cas_predits_semaine_suivante"})
    latest["score_global_predictif"] = minmax_series(latest["cas_predits_semaine_suivante"])

    return latest[[
        "code_departement", "nom_departement", "annee_semaine",
        "cas_predits_semaine_suivante", "tendance_evolution_cas", "score_global_predictif"
    ]]

# -----------------------------
# CHARGEMENT fichiers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_csv_any(path: Path, sep_guess: str = ";") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, sep=sep_guess)
    except Exception:
        # fallback sans sep
        df = pd.read_csv(path)
    df = coerce_schema(df)
    return df

# -----------------------------
# UI — SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Paramètres")

vue = st.sidebar.radio(
    "Vue",
    options=("🇫🇷 France métropolitaine", "🟣 Région Grand Est"),
    index=0,
    help="Bascule entre la vue nationale et la vue Grand Est."
)

engine = st.sidebar.radio(
    "Moteur prédictif",
    options=("⚡ MVP rapide (rolling 3w)", "🧠 Prophet (plus précis)"),
    index=0,
    help="Si Prophet échoue sur ta machine (ex: tbb.dll), l'app repasse automatiquement en MVP."
)

st.sidebar.markdown("---")
if vue.startswith("🇫🇷"):
    default_csv = str(NATIONAL_CSV)
else:
    default_csv = str(REGIONAL_CSV)

csv_path = st.sidebar.text_input("Chemin du CSV analytique", value=default_csv)
st.sidebar.caption("Par défaut : data/clean-data/donnees_analytiques_*.csv")

# -----------------------------
# DATA + PREDICTIONS
# -----------------------------
def run_engine(vue_label: str, engine_label: str, csv_path_str: str):
    """
    Retourne :
      df_full     -> données historiques (coercées)
      df_display  -> dernières valeurs par département avec colonnes prédictives
      last_week   -> libellé de semaine au plus récent
      used_engine -> 'prophet' ou 'mvp'
    """
    csv_path = Path(csv_path_str)
    df_hist = load_csv_any(csv_path)
    if df_hist.empty:
        return df_hist, pd.DataFrame(), None, "none"

    # Filtrage métropole côté national : conf déléguée aux modules Prophet (ils le font déjà).
    # Ici, pour MVP on garde tel quel.

    # Mode Prophet
    if engine_label.startswith("🧠") and (app_national or app_regional):
        try:
            if vue_label.startswith("🇫🇷"):
                # National
                if not app_national:
                    raise RuntimeError("Module app_national indisponible")
                df_loaded = app_national.charger_donnees(csv_path_str)  # gère zfill et Corse
                if df_loaded.empty:
                    raise RuntimeError("Chargement national vide")
                df_pred = app_national.entrainer_et_predire(df_loaded)
                df_display = app_national.calculer_score(df_pred)
                last_week = str(df_display["annee_semaine"].iloc[0])
                return df_loaded, df_display, last_week, "prophet"
            else:
                # Grand Est
                if not app_regional:
                    raise RuntimeError("Module app (régional) indisponible")
                df_loaded = app_regional.charger_donnees(csv_path_str)
                if df_loaded.empty:
                    raise RuntimeError("Chargement régional vide")
                df_pred = app_regional.entrainer_et_predire(df_loaded)
                df_display = app_regional.calculer_score(df_pred)
                last_week = str(df_display["annee_semaine"].iloc[0])
                return df_loaded, df_display, last_week, "prophet"
        except Exception as e:
            st.warning(
                f"Le moteur Prophet a échoué ({e!s}).\n"
                "→ Repli automatique sur le **MVP rapide**. "
                "Astuce Windows : lance `fix_tbb.bat` en admin si l’erreur évoque `tbb.dll`."
            )

    # MVP fallback (rapide)
    # Pour l’affichage carte, on veut 1 ligne / département (dernière semaine + indicateurs)
    df_pred_light = compute_light_predictions(df_hist)
    # On récupère aussi la dernière semaine présente dans l’historique (pour le sous-titre)
    df_hist["_week_date"] = df_hist["annee_semaine"].apply(week_to_datetime)
    last_date = df_hist["_week_date"].dropna().max()
    last_week = None
    if pd.notna(last_date):
        # retrouver le libellé original
        last_week = df_hist.loc[df_hist["_week_date"].idxmax(), "annee_semaine"]

    return df_hist, df_pred_light, str(last_week) if last_week else None, "mvp"

df_full, df_display, last_week_label, used_engine = run_engine(vue, engine, csv_path)

st.title("🌡️ Thermomètre Grippal Prédictif — Vue unifiée")

if df_full.empty or df_display.empty:
    st.error("Aucune donnée exploitable. Vérifie le chemin du CSV et le contenu attendu.")
    st.stop()

# Vérif colonnes minimales dans l’historique (affecte l’onglet 2)
missing = ensure_required_columns(df_full)
if missing:
    st.warning(f"Colonnes manquantes dans le CSV : {', '.join(missing)}")

# GeoJSON (prefer modules)
geojson = load_geojson_from_models()
if geojson is None:
    # fallback: fichier local si présent
    geo_path = Path("data/geo/departements.geojson")
    if geo_path.exists():
        with open(geo_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🗺️ Carte & KPIs", "🏥 Analyse département", "ℹ️ À propos"])

# --- TAB 1 : CARTE ---
with tab1:
    # En-tête + contexte
    label_vue = "France métropolitaine" if vue.startswith("🇫🇷") else "Région Grand Est"
    st.subheader(f"Carte de la tension — {label_vue} — semaine à venir (S+1)")
    if last_week_label:
        st.caption(f"Dernière semaine historique prise en compte : **{last_week_label}** · Moteur utilisé : **{used_engine.upper()}**")

    # KPIs globaux
    k1, k2, k3 = st.columns(3)
    k1.metric("Score moyen", f"{df_display['score_global_predictif'].mean():.2f}")
    k2.metric("Cas prédits (S+1)", f"{int(np.nan_to_num(df_display['cas_predits_semaine_suivante']).sum())}")
    k3.metric("Tendance moyenne", f"{np.nanmean(df_display['tendance_evolution_cas']):+.1%}")

    # Carte
    if geojson is None:
        st.warning("Fond de carte GeoJSON introuvable. Fourni `data/geo/departements.geojson` (feature.properties.code).")
    else:
        # Centre/zoom selon la vue
        if vue.startswith("🇫🇷"):
            zoom = 4.6
            center = {"lat": 46.6, "lon": 2.4}
        else:
            zoom = 6.5
            center = {"lat": 48.6921, "lon": 6.1844}

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
            Le **moteur** peut être *MVP rapide* (rolling 3 semaines) ou *Prophet* (plus robuste, mais plus lourd).
            """
        )

# --- TAB 2 : ANALYSE DEPARTEMENT ---
with tab2:
    st.subheader("Analyse détaillée par département")
    # Liste départements issue de df_display (1 ligne/dep)
    dep_options = sorted(df_display["nom_departement"].dropna().unique().tolist())
    dep_name = st.selectbox("Choisir un département", dep_options)

    dep_code = df_display.loc[df_display["nom_departement"] == dep_name, "code_departement"].iloc[0]

    # Historique pour ce département (df_full)
    df_hist_dep = df_full[df_full["code_departement"] == dep_code].copy()
    df_hist_dep["_week_date"] = df_hist_dep["annee_semaine"].apply(week_to_datetime)
    df_hist_dep = df_hist_dep.sort_values("_week_date")

    # Indicateurs actuels (depuis df_display)
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

# --- TAB 3 : A PROPOS ---
with tab3:
    st.markdown(
        """
        ### Objectif
        Anticiper les **zones de tension** liées à la grippe pour **aider l’allocation** des vaccins et ressources.

        ### Deux moteurs prédictifs
        - **⚡ MVP rapide** : moyenne glissante 3 semaines (fallback robuste et instantané).
        - **🧠 Prophet** : modélisation séries temporelles (saisonnalités) — plus précis.
          - Sous Windows, si une erreur **`tbb.dll`** apparaît, lance `fix_tbb.bat` en administrateur (fourni au repo).

        ### Données attendues
        CSV analytique par département / semaine, colonnes min :
        `code_departement`, `nom_departement`, `annee_semaine`, `total_cas_semaine`.
        Les colonnes facultatives (si présentes) sont utilisées telles quelles :
        `cas_predits_semaine_suivante`, `tendance_evolution_cas`, `score_global_predictif`.

        ### Conseils
        - Vue **France** : utilise `data/clean-data/donnees_analytiques_france.csv`
        - Vue **Grand Est** : utilise `data/clean-data/donnees_analytiques_grand_est.csv`
        - Fond de carte : auto (modules), sinon `data/geo/departements.geojson` (clé `feature.properties.code`).
        """
    )
