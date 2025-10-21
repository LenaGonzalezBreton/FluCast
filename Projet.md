# Rapport Détaillé : Optimisation de la Stratégie Vaccinale contre la Grippe

Ce rapport présente un plan d'action complet et amélioré pour le développement d'un système prédictif d'optimisation de la stratégie vaccinale contre la grippe, utilisant Prophet et Streamlit. Le projet vise à fournir aux décideurs de santé publique et aux professionnels de terrain des outils d'aide à la décision basés sur des données ouvertes et des modèles prédictifs avancés.

## Vue d'Ensemble du Projet

Le projet s'inscrit dans le cadre d'un hackathon Epitech visant à exploiter les données publiques ouvertes pour résoudre un défi majeur de santé publique[^1]. Les épidémies de grippe saisonnière représentent une charge importante pour le système de santé français, nécessitant une planification rigoureuse des campagnes de vaccination et une gestion optimale des ressources médicales. L'approche proposée transcende la simple prédiction numérique pour créer une véritable intelligence stratégique permettant d'anticiper les besoins et d'optimiser l'allocation des ressources[^2].

### Objectifs Stratégiques

Le projet poursuit quatre objectifs principaux intimement liés. Premièrement, développer des modèles prédictifs robustes capables d'estimer avec précision les besoins en vaccins en analysant les tendances historiques de couverture vaccinale et les indicateurs avancés sanitaires. Deuxièmement, optimiser la distribution des vaccins en pharmacie en tenant compte des données de distribution réelles et des actes de vaccination effectués. Troisièmement, anticiper les passages aux urgences et les actes SOS Médecins liés à la grippe afin de préparer les établissements de santé aux pics épidémiques. Quatrièmement, améliorer l'accès aux soins en identifiant les zones géographiques où les couvertures vaccinales sont insuffisantes et en proposant des stratégies ciblées[^1].

### Approche Innovante : Le Score de Tension Vaccinale

L'élément différenciateur de ce projet réside dans le concept de Score de Tension Vaccinale prédictif[^2][^3]. Plutôt que de simplement prédire un volume de vaccins nécessaire, le système calcule un indice composite multidimensionnel qui évalue le niveau de risque épidémique pour chaque zone géographique. Cette approche permet de transformer les données brutes en recommandations actionnables pour les décideurs.

Le Score de Tension Vaccinale (S_tv) se calcule selon la formule suivante : $S_{tv} = w_1 \cdot D + w_2 \cdot E + w_3 \cdot L$[^2][^4]

Chaque composante capture une dimension spécifique du risque. La composante D (Risque Démographique) quantifie la vulnérabilité intrinsèque d'une population basée sur sa structure d'âge, sa densité et sa croissance. La composante E (Pression Épidémiologique) mesure l'intensité actuelle et prédite de l'activité grippale, intégrant les passages aux urgences, les consultations médicales et les signaux faibles comme les tendances de recherche en ligne. La composante L (Vulnérabilité Logistique et Comportementale) évalue la capacité du système de santé local à répondre à la demande vaccinale, tenant compte de la couverture vaccinale historique et de la densité de l'offre de soins[^4].

## Architecture Technique et Technologies

### Stack Technologique

Le projet s'appuie sur un écosystème technologique Python robuste et éprouvé[^2][^5]. Prophet, développé par Facebook (Meta), constitue le cœur du moteur prédictif. Cette bibliothèque excelle dans l'analyse de séries temporelles avec saisonnalité, gérant automatiquement les variations hebdomadaires et annuelles caractéristiques des épidémies grippales[^3][^6][^7]. Prophet permet également d'intégrer des régresseurs externes (météo, pollution, tendances comportementales) pour enrichir les prédictions[^8].

Streamlit sert de framework pour le développement rapide du dashboard interactif[^2][^9]. Cette bibliothèque permet de créer des applications web de data science entièrement en Python, sans nécessiter de connaissances en développement web frontend. Les avantages incluent un déploiement rapide, une mise à jour en temps réel et une interactivité native avec les visualisations[^10][^11][^12].

L'écosystème de data science Python standard complète la stack avec Pandas pour la manipulation de données, NumPy pour les calculs numériques, et Scikit-learn pour le prétraitement et la validation. Pour les modèles complémentaires, XGBoost et LightGBM offrent des capacités de gradient boosting performantes sur données tabulaires[^2]. Enfin, Folium et Plotly Express permettent de créer des visualisations cartographiques interactives essentielles pour la représentation géographique des scores de tension[^2].

### Architecture Modulaire

Le système s'organise en modules distincts assurant la séparation des préoccupations et la maintenabilité du code. Le module de collecte et préparation des données automatise le téléchargement et la standardisation des multiples sources de données. Le pipeline de nettoyage et d'intégration traite les valeurs manquantes, les doublons et les anomalies, puis fusionne les données sur les codes géographiques INSEE[^13][^14][^15].

Le moteur de feature engineering transforme les données brutes en variables prédictives pertinentes, calculant les indicateurs démographiques, épidémiologiques et logistiques normalisés[^4]. Le module de modélisation Prophet génère les prédictions de passages aux urgences et calcule les scores de tension. Enfin, l'application Streamlit orchestre l'ensemble, offrant une interface utilisateur intuitive pour la visualisation et l'exploration des résultats[^9][^16].

## Sources de Données : Enrichissement Stratégique

### Datasets Fondamentaux

Le projet s'appuie sur un socle de données officielles de haute qualité. Les données IQVIA fournissent des informations précises sur la distribution de vaccins et les actes de vaccination en pharmacie au niveau départemental[^1][^17]. L'Indicateur Avancé Sanitaire (IAS) de Santé publique France offre un signal précoce sur les tendances de vaccination à l'échelle régionale[^1][^5].

Les données de couvertures vaccinales, disponibles aux niveaux national, régional et départemental, permettent d'analyser les tendances historiques et d'identifier les zones sous-vaccinées[^1][^5]. Les passages aux urgences et actes SOS Médecins pour grippe et syndrome grippal, collectés par le réseau OSCOUR, constituent la variable cible principale pour les prédictions Prophet[^1][^5][^18]. En 2024-2025, près de 700 structures d'urgences participent au réseau, enregistrant 96% des passages aux urgences nationaux[^18].

Le réseau Sentinelles complète la surveillance avec des données hebdomadaires sur les consultations pour infections respiratoires aiguës en médecine de ville[^5][^19]. Durant la saison 2024-2025, l'épidémie a généré près de 3 millions de consultations et 30 000 hospitalisations après passage aux urgences, démontrant l'ampleur du phénomène[^20].

### Enrichissement Démographique et Socio-économique

Les données INSEE constituent un pilier essentiel pour la composante D du score[^4]. La population par tranche d'âge au niveau communal et départemental permet de calculer précisément la proportion de personnes de plus de 65 ans, population prioritaire pour la vaccination antigrippale. La densité de population au km² influence directement la vitesse de propagation virale dans les zones densément peuplées.

Les indicateurs de précarité (taux de pauvreté, revenus médians) sont corrélés avec l'accès aux soins et la couverture vaccinale[^4][^21]. En France, la prévalence du diabète atteint 6,48% en 2023, avec de fortes disparités régionales[^21]. Ces populations présentant des comorbidités constituent des groupes à haut risque nécessitant une priorisation vaccinale.

### Innovation : Facteurs Environnementaux et Comportementaux

L'enrichissement majeur du projet réside dans l'intégration de données environnementales et comportementales. Les données climatologiques de Météo-France, gratuitement accessibles depuis 2024, incluent les températures, l'humidité relative et les précipitations au niveau quotidien[^22][^23]. Ces facteurs météorologiques influencent significativement la transmission virale et la survie du virus grippal dans l'environnement.

Les données de qualité de l'air du Laboratoire Central de Surveillance de la Qualité de l'Air (LCSQA) fournissent les concentrations horaires de polluants atmosphériques (PM10, PM2.5, NO2, O3)[^24][^25]. La pollution particulaire aggrave les symptômes respiratoires et pourrait moduler la susceptibilité aux infections grippales.

Google Trends offre un signal comportemental unique, capturant en temps quasi réel l'intérêt de recherche du public pour des termes liés à la grippe ("grippe", "symptômes grippe", "fièvre", "vaccin grippe")[^4]. Ces données peuvent servir d'indicateur avancé de l'arrivée de la vague épidémique, détectant les premiers frémissements avant même que les cas ne se présentent aux urgences[^2].

### Données sur l'Offre de Soins

La densité de l'offre de soins structure la composante L du score. Le Répertoire Partagé des Professionnels de Santé (RPPS) recense la démographie des médecins par département, permettant de calculer le ratio médecins généralistes pour 10 000 habitants[^5]. Les données OpenStreetMap sur la localisation des pharmacies permettent d'analyser l'accessibilité géographique aux points de vaccination[^5].

Les données d'hospitalisation de l'Agence Technique de l'Information sur l'Hospitalisation (ATIH) documentent la capacité en lits et les flux de patients dans les établissements[^26][^27]. En 2023, ces données offrent une vision globale de l'activité des 4 champs du Programme de Médicalisation des Systèmes d'Information (PMSI).

## Plan de Réalisation Détaillé

### Phase 1 : Collecte et Préparation des Données (8 heures)

La première phase établit les fondations du projet en rassemblant l'ensemble des données nécessaires. Les deux membres de l'équipe data commencent par télécharger les datasets officiels de base : IQVIA, IAS, couvertures vaccinales aux trois niveaux géographiques, passages urgences et SOS Médecins, et données Sentiweb (2 heures). Parallèlement, ils collectent les données démographiques INSEE incluant la population par âge, la densité et les indicateurs de précarité aux niveaux départemental et communal (1,5 heure).

L'enrichissement du projet nécessite l'acquisition de datasets supplémentaires : données climatologiques quotidiennes de Météo-France (température, humidité, précipitations), données de qualité de l'air du LCSQA (PM10, PM2.5, NO2), localisation des pharmacies et hôpitaux via OpenStreetMap, et démographie des médecins du RPPS (2 heures)[^22][^24][^23][^25]. L'extraction des tendances Google Trends pour les mots-clés sanitaires pertinents par région complète cette collecte (1 heure).

La standardisation des formats constitue une étape critique souvent sous-estimée[^4]. L'équipe harmonise les formats de dates selon la norme ISO 8601, uniformise les codes géographiques INSEE, assure un encodage UTF-8 cohérent et standardise les séparateurs CSV (1,5 heure). Cette normalisation préventive évite de nombreux problèmes lors des phases ultérieures.

### Phase 2 : Nettoyage et Intégration (12 heures)

Le pipeline de nettoyage automatisé représente l'investissement le plus rentable du projet[^13][^14][^15]. L'équipe développe des scripts Python réutilisables exploitant les capacités de Pandas pour détecter et supprimer les doublons, gérer intelligemment les valeurs manquantes par imputation statistique ou prédictive, corriger automatiquement les types de données et normaliser les chaînes de caractères (3 heures)[^28][^15].

La gestion des valeurs aberrantes combine méthodes statistiques et validation métier (2 heures). Les techniques statistiques classiques (Interquartile Range, Z-score) détectent les outliers numériques, mais la validation métier reste essentielle pour identifier les anomalies logiques (par exemple, un taux de vaccination supérieur à 100% ou des températures incohérentes avec la saison).

La fusion des données géographiques constitue le cœur de l'intégration (3 heures)[^4]. Tous les datasets sont joints via les codes géographiques INSEE standardisés, créant une table de référence unifiée avec la hiérarchie complète commune → département → région. Cette structure permet des analyses à granularité variable selon les besoins.

L'agrégation temporelle aligne toutes les séries sur une granularité hebdomadaire, cohérente avec les cycles épidémiques de la grippe (2 heures)[^4]. Les features calendaires enrichissent les données : semaine épidémiologique (S01-S52), mois, saison, jours fériés et vacances scolaires qui influencent les comportements et la transmission.

La validation de la qualité données via tests automatisés garantit la fiabilité des analyses ultérieures (2 heures). Les tests vérifient l'intégrité référentielle entre tables, la cohérence temporelle des séries, les plages de valeurs attendues pour chaque variable et la complétude par période et zone géographique. Un rapport de qualité détaillé documente les éventuels problèmes résiduels.

### Phase 3 : Feature Engineering (8 heures)

Le feature engineering transforme les données brutes en variables prédictives exploitables par les modèles[^4]. Pour la composante démographique D, l'équipe calcule le ratio de population de plus de 65 ans, la densité de population, l'indice de vieillissement, le taux de croissance démographique et l'indice de précarité, tous normalisés sur l'échelle [^1] (2 heures).

Les features épidémiologiques capturent la dynamique de l'épidémie (2 heures). Les taux de passages aux urgences pour 100 000 habitants permettent les comparaisons entre zones de tailles différentes. Les variations hebdomadaires et mensuelles révèlent les tendances. L'IAS normalisé, les seuils épidémiques régionaux et les trends Google Trends standardisés complètent cette composante E.

Les features logistiques et comportementales quantifient la capacité du système de santé (1,5 heure). La densité de pharmacies et médecins pour 10 000 habitants, le taux de couverture vaccinale historique, le gap de vaccination (potentiel non exploité) et la capacité hospitalière disponible structurent la composante L du score.

L'intégration des features environnementales enrichit les modèles avec le contexte météorologique et de qualité de l'air (1,5 heure)[^22][^24]. Les températures moyennes, minimales et maximales, l'humidité relative, les précipitations et les indices de pollution (PM10, PM2.5, NO2) sont calculés en moyennes mobiles sur 7 et 14 jours pour lisser les variations quotidiennes.

Les features temporelles de type lag et rolling complètent l'arsenal prédictif (1 heure). Les décalages temporels (lags) de 1 à 4 semaines pour les variables clés permettent au modèle de capturer la mémoire du système. Les moyennes et écarts-types glissants sur 2 à 8 semaines, ainsi que les taux de croissance semaine sur semaine, révèlent les dynamiques sous-jacentes.

### Phase 4 : Modélisation Prédictive (12 heures)

La modélisation avec Prophet commence par la préparation des données dans le format requis (2 heures)[^3][^29][^7]. Chaque série temporelle est structurée avec les colonnes obligatoires 'ds' (date au format datetime) et 'y' (variable à prédire). Des datasets séparés sont créés pour chaque département permettant une modélisation granulaire. Les jours fériés français et les périodes de vacances scolaires sont intégrés comme événements spéciaux.

Le modèle Prophet baseline établit une première référence (3 heures)[^30][^7]. Configuré avec un mode de saisonnalité multiplicatif adapté aux épidémies, il intègre les saisonnalités hebdomadaires et annuelles automatiquement détectées. La validation croisée teste la robustesse des prédictions sur différentes fenêtres temporelles, et les métriques de performance (MAE, RMSE, MAPE) sont calculées.

L'enrichissement du modèle Prophet avec des régresseurs externes exploite tout le potentiel du feature engineering (3 heures)[^8]. Les variables météorologiques (température, humidité), les indices de pollution, les tendances Google Trends et les taux de vaccination historiques sont intégrés comme covariables. Le tuning des hyperparamètres (changepoint_prior_scale contrôlant la flexibilité de la tendance, seasonality_prior_scale régulant l'amplitude des saisonnalités) optimise les performances[^8].

Le calcul du Score de Tension Vaccinale synthétise les trois composantes (2 heures)[^4]. La formule $S_{tv} = w_1 \cdot D + w_2 \cdot E + w_3 \cdot L$ est implémentée avec D représentant les features démographiques normalisées, E intégrant les prédictions Prophet normalisées, et L agrégeant les features logistiques normalisées. L'optimisation des poids $w_1, w_2, w_3$ peut se faire par calibration sur données historiques ou par expertise métier (démarrer avec des poids égaux de 0.33).

La validation finale et le calcul des métriques assurent la qualité du système prédictif (2 heures). Le backtesting sur 2-3 saisons grippales passées teste la capacité de généralisation. L'analyse des résidus révèle les biais systématiques éventuels. Les intervalles de confiance de Prophet quantifient l'incertitude des prédictions, information cruciale pour la prise de décision.

### Phase 5 : Dashboard Streamlit (12 heures)

Le développement de l'application Streamlit transforme les modèles en outil utilisable (2 heures)[^9][^31][^12]. L'architecture modulaire sépare les responsabilités : app.py pour l'orchestration principale, models.py pour les fonctions de modélisation, viz.py pour les visualisations et utils.py pour les utilitaires. La page configuration définit le titre, l'icône et le layout. Une sidebar propose des filtres interactifs sur la région, le département et la période temporelle.

La carte interactive France constitue le cœur de la visualisation (3 heures). Développée avec Folium ou Plotly, la carte choroplèthe colorie chaque département selon son Score de Tension Vaccinale en vert (faible), orange (moyen) ou rouge (élevé). Les tooltips affichent les détails au survol : score exact, composantes D/E/L, prédictions pour S+1. La navigation permet de zoomer et de basculer entre les vues nationales, régionales et départementales.

Les graphiques temporels racontent l'histoire de l'épidémie (3 heures). L'évolution historique des passages aux urgences est représentée avec les prédictions Prophet pour S+1 et S+2, incluant les intervalles de confiance. L'évolution du score de tension dans le temps révèle les zones entrant en phase critique. Les comparaisons inter-départements facilitent les analyses comparatives. La décomposition Prophet (tendance, saisonnalité hebdomadaire et annuelle) permet de comprendre les mécanismes sous-jacents[^7].

Les indicateurs clés de performance (KPIs) synthétisent l'information essentielle (1,5 heure). Des cartes métriques affichent le score de tension actuel, la prédiction pour S+1, la variation par rapport à la semaine précédente (flèche verte ou rouge), le taux de couverture vaccinale actuel, la capacité hospitalière disponible et le top 5 des départements à risque nécessitant une attention prioritaire.

Les fonctionnalités avancées enrichissent l'expérience utilisateur (2,5 heures). Les filtres interactifs permettent de sélectionner dynamiquement les dates, les zones géographiques et les populations d'intérêt. L'export des données en CSV ou Excel facilite les analyses complémentaires. Les tooltips explicatifs et l'aide contextuelle rendent l'outil accessible aux non-experts. Le mode comparaison multi-zones permet de benchmarker plusieurs départements simultanément. Les alertes automatiques signalent les dépassements de seuils critiques nécessitant une action immédiate.

### Phase 6 : Tests et Documentation (6 heures)

Les tests unitaires garantissent la fiabilité des composants individuels (2 heures). Utilisant le framework pytest, l'équipe teste les fonctions de nettoyage de données, les calculs de features, les prédictions Prophet et le calcul des scores. L'objectif de couverture de code supérieur à 80% assure une robustesse satisfaisante.

Les tests d'intégration valident le fonctionnement du système complet (1,5 heure). Le pipeline entier depuis les données brutes jusqu'aux prédictions et au dashboard est exécuté end-to-end. La gestion des erreurs (fichiers manquants, formats incorrects) est testée. Les performances (temps de calcul pour différentes tailles de données) sont mesurées. La compatibilité avec différents formats de données d'entrée est vérifiée.

La documentation du code assure la maintenabilité et la transmissibilité du projet (1,5 heure). Toutes les fonctions sont documentées avec des docstrings au format Google ou NumPy. Un README.md complet guide l'installation, l'utilisation et fournit des exemples. Les sections de code complexes incluent des commentaires explicatifs. Un schéma d'architecture visualise l'organisation des composants.

Le guide utilisateur rend l'outil accessible aux utilisateurs finaux (1 heure). Ce document couvre l'installation de l'environnement Python et des dépendances, le téléchargement et la préparation des datasets, le lancement du pipeline de traitement, l'utilisation du dashboard Streamlit, l'interprétation des résultats et des scores, et une FAQ répondant aux questions courantes.

## Améliorations Clés par Rapport aux Rapports Initiaux

Le présent rapport améliore substantiellement les documents de travail initiaux sur plusieurs dimensions critiques[^2][^3][^4]. L'enrichissement des datasets constitue l'amélioration la plus significative, passant de 11 à 16 sources de données. L'ajout des données climatologiques de Météo-France permet d'intégrer l'influence de la température et de l'humidité sur la transmission virale[^22][^23]. Les données de qualité de l'air du LCSQA ajoutent la dimension de pollution atmosphérique[^24][^25]. Google Trends apporte un signal comportemental unique capturant l'intérêt public en temps quasi réel[^4].

L'automatisation du nettoyage et de la préparation des données transforme des processus manuels chronophages en pipelines reproductibles[^13][^14][^15]. Les scripts Python automatisés détectent et corrigent systématiquement les problèmes de qualité, réduisant drastiquement le temps consacré à ces tâches répétitives et minimisant les erreurs humaines.

Le modèle Prophet enrichi avec regressors externes exploite pleinement les données collectées[^8]. Alors que les rapports initiaux proposaient un Prophet basique, la version améliorée intègre les facteurs météorologiques, de pollution, comportementaux et logistiques comme covariables, augmentant significativement la précision des prédictions.

La granularité géographique améliorée permet désormais de travailler au niveau communal pour certaines analyses, en plus des niveaux départemental et régional[^4]. Cette flexibilité permet d'adapter la finesse de l'analyse aux besoins spécifiques et à la disponibilité des données.

Le plan d'action opérationnel détaillé avec 29 tâches structurées en 6 phases offre une feuille de route claire pour l'exécution. Chaque tâche spécifie la durée estimée, les dépendances, le responsable et les datasets requis, facilitant la coordination d'équipe et le suivi de l'avancement.

Les métriques et validation robustes incluent désormais le backtesting sur plusieurs saisons, l'analyse des résidus et l'évaluation des intervalles de confiance, garantissant la fiabilité des prédictions avant déploiement.

## Considérations Pratiques pour l'Implémentation

### Gestion du Temps et Priorisation

Avec une durée totale estimée à 58 heures de travail (7,2 jours à 8 heures par jour), le projet s'inscrit dans un cadre de hackathon de 48 heures pour une équipe de 5 personnes (240 personnes-heures disponibles). Le plan doit être exécuté avec rigueur et les tâches doivent être parallélisées efficacement.

La priorisation des datasets et fonctionnalités est cruciale. Les 8 datasets de priorité haute (IQVIA, IAS, couvertures vaccinales, urgences, INSEE démographie, Météo-France, Google Trends) doivent être traités en premier. Les 6 datasets de priorité moyenne peuvent être intégrés si le temps le permet. Les 2 datasets de priorité basse (ATIH hospitalisation, comorbidités détaillées) sont optionnels.

### Défis Techniques et Solutions

L'hétérogénéité des formats de données représente un défi majeur. Les solutions incluent la création de parsers spécifiques pour chaque source, l'utilisation de bibliothèques robustes (Pandas avec gestion d'erreurs), et la documentation précise des transformations appliquées.

La taille des datasets climatologiques et de pollution peut poser des problèmes de performance. Les stratégies d'optimisation comprennent le filtrage temporel (ne charger que les années pertinentes), l'agrégation spatiale (moyennes départementales plutôt que données par station), et le calcul incrémental avec sauvegarde d'états intermédiaires.

La validation de la qualité des prédictions Prophet nécessite une expertise métier. La collaboration avec des épidémiologistes permet de valider la plausibilité des résultats, la calibration des hyperparamètres basée sur la connaissance du domaine, et l'interprétation correcte des intervalles de confiance.

### Scalabilité et Évolutions Futures

Le système doit être conçu pour évoluer au-delà du MVP initial. L'extension à d'autres régions françaises nécessite simplement l'ajout des codes géographiques correspondants, la structure étant déjà modulaire. L'intégration de datasets supplémentaires (réseaux sociaux, mobilité, variants viraux) peut enrichir progressivement les prédictions. L'ajout de nouveaux modèles (LSTM pour capturer les dépendances temporelles longues, modèles d'ensemble combinant Prophet, XGBoost et réseaux neuronaux) peut améliorer les performances.

Le déploiement en production requiert des considérations spécifiques. L'hébergement sur Streamlit Cloud ou une infrastructure cloud (AWS, GCP, Azure) assure la disponibilité. L'actualisation automatique des données via des jobs planifiés (Airflow, cron) maintient la fraîcheur des informations. Le monitoring des performances et la détection d'anomalies garantissent la fiabilité opérationnelle. La sécurisation de l'accès pour les utilisateurs professionnels protège la confidentialité.

## Conclusion et Perspectives

Ce projet représente une avancée significative dans l'exploitation des données ouvertes pour la santé publique. En combinant des sources de données traditionnelles (surveillance épidémiologique, démographie) avec des signaux innovants (météo, pollution, tendances comportementales), le système offre une vision holistique du risque épidémique.

L'approche centrée sur le Score de Tension Vaccinale transforme des données complexes en un indicateur actionnable pour les décideurs[^2][^3][^4]. Plutôt que de noyer les utilisateurs sous des dizaines de graphiques, le système hiérarchise l'information et guide l'action vers les zones nécessitant une intervention prioritaire.

L'utilisation de Prophet et Streamlit permet un développement rapide sans sacrifier la sophistication analytique[^6][^9][^12]. Ces technologies modernes, bien documentées et disposant d'importantes communautés, facilitent la maintenance et l'évolution du système.

Les 16 datasets intégrés offrent une granularité et une richesse d'information sans précédent pour ce type de projet. L'automatisation du pipeline de traitement permet d'actualiser régulièrement les analyses sans intervention manuelle intensive, condition essentielle pour un déploiement opérationnel.

Le plan détaillé en 29 tâches fournit une feuille de route claire pour l'exécution. Avec une équipe disciplinée et une bonne coordination, le MVP peut être réalisé dans le cadre du hackathon de 48 heures, démontrant la faisabilité de l'approche.

Au-delà du hackathon, ce projet pose les bases d'un véritable outil d'aide à la décision pour les Agences Régionales de Santé, les pharmaciens et les professionnels de santé. L'extension à d'autres pathologies saisonnières (bronchiolite, gastro-entérite) ou à d'autres dimensions de la santé publique (déserts médicaux, accès aux soins) est envisageable avec la même architecture.

En définitive, ce projet illustre le potentiel transformateur des données ouvertes et de l'intelligence artificielle au service de la santé publique, permettant de passer d'une gestion réactive des crises sanitaires à une approche véritablement anticipative et optimisée.
<span style="display:none">[^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46][^47][^48][^49][^50][^51][^52][^53][^54][^55][^56][^57][^58][^59][^60][^61][^62][^63][^64][^65]</span>

<div align="center">⁂</div>

[^1]: project-1.pdf

[^2]: Contexte.md

[^3]: Prediction.md

[^4]: Calculs.md

[^5]: Datasets.md

[^6]: https://www.kaggle.com/code/prashant111/tutorial-time-series-forecasting-with-prophet

[^7]: https://www.geeksforgeeks.org/data-science/time-series-analysis-using-facebook-prophet/

[^8]: https://zerotomastery.io/blog/time-series-forecasting-with-facebook-prophet/

[^9]: https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/

[^10]: https://www.youtube.com/watch?v=p2pXpcXPoGk

[^11]: https://clickhouse.com/engineering-resources/python-dashboard-streamlit

[^12]: https://streamlit.io

[^13]: https://www.machinelearningmastery.com/automating-data-cleaning-processes-with-pandas/

[^14]: https://www.kdnuggets.com/creating-automated-data-cleaning-pipelines-using-python-and-pandas

[^15]: https://www.geeksforgeeks.org/data-analysis/how-to-automate-data-cleaning-in-python/

[^16]: https://www.freecodecamp.org/news/build-a-real-time-network-traffic-dashboard-with-python-and-streamlit/

[^17]: https://www.data.gouv.fr/datasets/vaccination-grippe-2024-2025/

[^18]: https://odisse.santepubliquefrance.fr/explore/dataset/grippe-passages-aux-urgences-et-actes-sos-medecins-france/?flg=fr-fr

[^19]: https://www.sentiweb.fr

[^20]: https://beh.santepubliquefrance.fr/beh/2025/17/2025_17_1.html

[^21]: https://www.federationdesdiabetiques.org/information/diabete/chiffres-france

[^22]: https://bonnespratiques-eau.fr/2024/01/12/les-donnees-meteo-france-accessibles-gratuitement/

[^23]: https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes/

[^24]: https://www.lcsqa.org/fr/les-donnees-nationales-de-qualite-de-lair

[^25]: https://www.data.gouv.fr/datasets/donnees-temps-reel-de-mesure-des-concentrations-de-polluants-atmospheriques-reglementes-1/

[^26]: https://www.atih.sante.fr/actualites/chiffres-cles-de-l-hospitalisation-2023

[^27]: https://www.atih.sante.fr/chiffres-cles-de-l-hospitalisation

[^28]: https://www.w3schools.com/python/pandas/pandas_cleaning.asp

[^29]: http://facebook.github.io/prophet/docs/quick_start.html

[^30]: https://www.youtube.com/watch?v=j0eioK5edqg

[^31]: https://www.flowt.fr/blog/deployer-dashboard-interactif-streamlit-tutoriel

[^32]: https://www.drias-climat.fr

[^33]: https://www.statistiques.developpement-durable.gouv.fr/pollution-de-lair-0

[^34]: https://www.data.gouv.fr/datasets/vaccination-grippe-2023-2024/

[^35]: https://www.data.gouv.fr/organizations/meteo-france/

[^36]: https://www.airparif.fr/surveiller-la-pollution/pollution-en-direct-en-ile-de-france

[^37]: https://donneespubliques.meteofrance.fr/?fond=rubrique\&id_rubrique=29

[^38]: https://www.prevair.org

[^39]: https://www.snds.gouv.fr/SNDS/Open-Data

[^40]: http://archives-climat.fr

[^41]: https://www.atmo-france.org/article/les-donnees-air-disponibles

[^42]: https://www.has-sante.fr/jcms/p_3604478/fr/vaccination-contre-la-grippe-saisonniere-des-personnes-de-65-ans-et-plus-place-des-vaccins-efluelda-et-fluad-recommandation

[^43]: https://donneespubliques.meteofrance.fr

[^44]: https://www.rcafactory.com/les-10-tendances-social-media-sante-et-pharma-2025-par-rca-factory/

[^45]: https://drees.solidarites-sante.gouv.fr/publications-communique-de-presse/panoramas-de-la-drees/240718_Panorama_EtablissementsSante2024

[^46]: https://sante.gouv.fr/IMG/pdf/referenciel_pratiques_diabete.pdf

[^47]: https://www.cbnews.fr/cb/image-10-tendances-social-media-sante-pharma-2025-rca-factory-89637

[^48]: https://www.vidal.fr/maladies/recommandations/diabete-de-type-2-1440.html

[^49]: https://kdsante.com/tendances-social-media-sante/

[^50]: https://www.data.gouv.fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/

[^51]: https://www.priseenchargepreventive.fr/diabete-type-2-maladie-chronique/

[^52]: https://fr.webedia-group.com/news/bien-etre/webedia-care-devoile-les-resultats-inedits-de-son-etude-sur-la-sante-et-le-bien-etre-des-18-34-ans-sur-les-reseaux-sociaux

[^53]: https://www.drees.solidarites-sante.gouv.fr/publications-communique-de-presse-documents-de-reference/panoramas-de-la-drees/250522_Panorama_etablissements-de-sante2025

[^54]: https://www.has-sante.fr/jcms/p_3636176/fr/parcours-de-soins-du-patient-adulte-vivant-avec-un-diabete-de-type-2-guide

[^55]: https://seomaniak.com/fr/comment-reseaux-sociaux-transforment-le-marketing-sante/

[^56]: https://www.scansante.fr

[^57]: https://www.santepubliquefrance.fr/les-actualites/2024/le-diabete-en-france-continue-de-progresser

[^58]: https://comarketing-news.fr/reseaux-sociaux-la-fatigue-numerique-sinstalle/

[^59]: https://geodes.santepubliquefrance.fr

[^60]: https://pasteur-lille.fr/centre-de-recherche/thematiques-de-recherche/diabete-et-obesite/

[^61]: https://www.youtube.com/watch?v=nE8897PrjLY

[^62]: https://www.youtube.com/watch?v=7ggkGBugbRI

[^63]: https://python.plainenglish.io/this-one-pandas-function-saved-me-hours-of-data-cleaning-77e45ca2fb2a

[^64]: https://www.tigerdata.com/blog/time-series-forecasting-with-timescaledb-and-prophet

[^65]: https://www.youtube.com/watch?v=asFqpMDSPdM

