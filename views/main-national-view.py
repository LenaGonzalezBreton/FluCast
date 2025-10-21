import sys
sys.path.append('../T-HAK-700-NCY_6')

import streamlit as st
import plotly.express as px

from models import app_national


# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    # MODIFIÉ : Titre de la page mis à jour
    page_title="Dashboard Grippe France",
    page_icon="🌡️",
    layout="wide"
)

# --- INTERFACE DE L'APPLICATION ---

# MODIFIÉ : Titre de l'application mis à jour
st.title("🌡️ Thermomètre Grippal Prédictif - France Métropolitaine")

# MODIFIÉ : Chemin vers le nouveau fichier de données national
df_historique = app_national.charger_donnees("data/clean-data/donnees_analytiques_france.csv")

if not df_historique.empty:
    df_avec_predictions = app_national.entrainer_et_predire(df_historique)
    df_final_pour_affichage = app_national.calculer_score(df_avec_predictions)

    # Création des onglets pour une navigation claire
    tab1, tab2, tab3 = st.tabs(["**Vue Nationale (Carte)**", "**Analyse par Département**", "**À Propos du Projet**"])

    # --- Onglet 1: Vue d'Ensemble ---
    with tab1:
        st.header(f"Carte de la Tension pour la semaine à venir (Prédiction)")
        st.info(
            """
            **Comment lire cette carte ?** Chaque département est coloré selon notre **score de tension prédictif**. 
            Une couleur plus foncée (rouge) indique une probabilité plus élevée de tension sur le système de santé la semaine prochaine.
            """
        )

        derniere_semaine_dispo = df_final_pour_affichage['annee_semaine'].iloc[0]
        st.subheader(f"Basé sur les données de la semaine : {derniere_semaine_dispo}")

        # Indicateurs clés pour toute la France
        col1, col2, col3 = st.columns(3)
        score_moyen = df_final_pour_affichage['score_global_predictif'].mean()
        cas_predits_total = df_final_pour_affichage['cas_predits_semaine_suivante'].sum()
        tendance_moyenne = df_final_pour_affichage['tendance_evolution_cas'].mean()

        col1.metric("Score de Tension Moyen National", f"{score_moyen:.2f}",
                    help="Le score moyen sur 1 pour la France. Plus il est proche de 1, plus le risque est élevé.")
        col2.metric("Total Cas Prédits (S+1)", f"{cas_predits_total}",
                    help="La somme des cas (urgences + SOS médecins) attendus pour la semaine prochaine, selon notre modèle.")
        col3.metric("Tendance Moyenne des Cas", f"{tendance_moyenne:+.1%}",
                    help="L'évolution moyenne des cas par rapport à la semaine précédente.")

        # Carte choroplèthe
        geojson = app_national.get_geojson()
        if geojson:
            fig = px.choropleth_mapbox(
                df_final_pour_affichage, geojson=geojson, locations='code_departement', featureidkey="properties.code",
                color='score_global_predictif', color_continuous_scale="YlOrRd", range_color=(0, 1),
                mapbox_style="carto-positron",
                # MODIFIÉ : Zoom et centre ajustés pour la France
                zoom=4.5,
                center={"lat": 46.8566, "lon": 2.3522},
                opacity=0.7,
                labels={'score_global_predictif': 'Score de Tension'}, hover_name='nom_departement',
                hover_data={'cas_predits_semaine_suivante': True, 'tendance_evolution_cas': ':.2%',
                            'code_departement': False}
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

    # --- Onglet 2: Analyse par Département ---
    with tab2:
        st.header("Analyse Détaillée par Département")
        departement_choisi = st.selectbox(
            "Sélectionner un département",
            options=sorted(df_final_pour_affichage['nom_departement'].unique())
        )

        df_departement = df_final_pour_affichage[df_final_pour_affichage['nom_departement'] == departement_choisi]

        if not df_departement.empty:
            st.subheader(f"Indicateurs pour : {departement_choisi}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Score de Tension", f"{df_departement['score_global_predictif'].iloc[0]:.2f}")
            col2.metric("Cas Prédits (S+1)", f"{df_departement['cas_predits_semaine_suivante'].iloc[0]}")
            col3.metric("Tendance des Cas", f"{df_departement['tendance_evolution_cas'].iloc[0]:+.1%}")

            st.subheader("Historique des cas hebdomadaires")
            st.info(
                "Ce graphique montre l'évolution historique des cas (urgences + SOS médecins) pour le département sélectionné.")

            df_historique_dep = df_historique[df_historique['nom_departement'] == departement_choisi]
            fig_hist = px.line(df_historique_dep, x='annee_semaine', y='total_cas_semaine',
                               title=f"Évolution des cas en {departement_choisi}",
                               labels={"annee_semaine": "Semaine", "total_cas_semaine": "Nombre total de cas"})
            fig_hist.update_traces(mode='lines+markers')
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- Onglet 3: À Propos ---
    with tab3:
        st.header("À Propos de Notre Projet")
        st.markdown("""
        Ce projet a été réalisé dans le cadre du **Hackathon Santé Datalab x EPITECH**.

        ### Objectif
        Développer un outil d'aide à la décision pour optimiser la stratégie vaccinale contre la grippe en **anticipant les zones de tension** sur le système de santé.

        ### Méthodologie
        Notre démarche se déroule en trois étapes clés :
        1.  **Préparation des Données :** Nous avons collecté, nettoyé et fusionné plusieurs jeux de données open-source (Santé publique France, INSEE) pour créer une base de données analytique complète par département et par semaine.
        2.  **Modélisation Prédictive :** Nous utilisons **Prophet**, un modèle de prédiction de séries temporelles, pour prévoir le nombre de cas de grippe (passages aux urgences et actes SOS Médecins) pour la semaine à venir. Le modèle est entraîné sur les données historiques pour apprendre la saisonnalité de l'épidémie.
        3.  **Scoring de Tension :** Un **score de tension** (de 0 à 1) est ensuite calculé pour chaque département. Il combine la **prédiction des cas futurs** avec des indicateurs de **vulnérabilité de la population**, comme le taux de couverture vaccinale de l'année précédente.

        ### L'Équipe
        * [Nom Membre 1]
        * [Nom Membre 2]
        * [Nom Membre 3]
        * [Nom Membre 4]
        * [Nom Membre 5]
        """)
else:
    st.warning(
        "Les données n'ont pas pu être chargées. Veuillez vérifier que le fichier 'clean-data/donnees_analytiques_france.csv' existe.")
