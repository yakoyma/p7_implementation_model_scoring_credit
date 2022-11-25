# Projet 7 : Implémentez un modèle de scoring
Parcours Data Scientist d'OpenClassrooms en partenariat avec CentraleSupélec.

L'objectif du projet consiste à implémenter un outil de "scoring crédit" (un dashboard interactif) pour calculer la probabilité qu’un client rembourse son crédit.
Il s'agit d'un outil d'aide à la décision d'octroi ou de refus de crédit à la consommation.

La source des données est la suivante : https://www.kaggle.com/c/home-credit-default-risk/data


## Les étapes du projet :
- Réalisation de l'analyse exploratoire ;
- Data preprocessing et définition des métriques d'évaluation des performances des modèles (utilisation de la librairie Sklearn pour les imputations, la transformation quantile et la standardisation des données) ; 
- Transfomation des données déséquilibrées : Over-sampling/Under-sampling (utilisation de la librairie Imblearn) ;
- Sélection des features pertinents (utilisation de la fonction RFECV de Sklearn) ; 
- Comparaison des différents modèles de classification par la validation croisée (utilisation de la fonction GridSearchCV de Sklearn) ;
- Définition de la fonction coût métier, de l'algorithme d'optimisation et la métrique d'évaluation (implémentation du F-beta score et utilisation du package Hyperopt) ;
- L'interprétabilité globale et locale du modèle (application de l'article 22 du RGPD).


## L'outil de scoring crédit (Tableau de bord ou Dashboard) :
- Réalisation de l'API Flask de prédiction du score déployée dans le cloud (Heroku) ;
- Réalisation d'un dashboard interactif (Streamlit) déployé dans le cloud (Streamlit share) ;
- Lien du dashboard : https://share.streamlit.io/yakoyma/p7-implementation-model-scoring-credit/main/dashboard.py
