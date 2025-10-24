# Analyse des campagnes marketing — Projet Camp_Market

Ce dépôt contient une exploration et une modélisation des données clients du jeu de données "Camp_Market". L'objectif principal est d'analyser le comportement des clients, segmenter la clientèle (clustering) et évaluer la performance prédictive des campagnes marketing historiques afin de recommander la meilleure campagne pour un client donné.

## Objectifs

- Nettoyage et enrichissement des données brutes.
- Analyse exploratoire (dépenses, canaux d'achat, cohorte, profils démographiques).
- Segmentation des clients via K-Means pour identifier des profils (ex. gros dépensiers, sensibles aux promotions).
- Construction de modèles prédictifs (Random Forest) pour estimer la probabilité d'acceptation de chaque campagne.
- Fournir des recommandations de campagne pour un client individuel.

## Données

- Fichier source brut : `Camp_Market.csv` (séparateur `;`).
- Fichier nettoyé utilisé dans les notebooks : `Camp_Market_cleaned.csv`.

Colonnes importantes (exemples) :
- ID, Year_Birth, Dt_Customer, Income
- MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
- NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumDealsPurchases, NumWebVisitsMonth
- AcceptedCmp1..5, Response

Le notebook effectue des opérations de remplissage de valeurs manquantes, calcul d'`Age`, `Tenure_Days`, `Total_Spending`, `Total_Purchases`, `Dependents`, etc., et exporte un fichier nettoyé `Camp_Market_cleaned.csv`.

## Fichiers et notebooks principaux

- `ML3.ipynb` : notebook principal contenant le pipeline complet (nettoyage, EDA, clustering, modélisation, prédictions). C'est le fichier principal dans ce dépôt.
- `Camp_Market.csv` : données brutes.
- `Camp_Market_cleaned.csv` : version nettoyée et enrichie utilisée pour l'analyse.
- `test.py` : script de test (si présent pour des essais rapides).

## Aperçu du pipeline

1. Prétraitement des données
   - Remplissage des valeurs manquantes (par ex. `Income`).
   - Transformation de `Dt_Customer` en date et calcul d'`Age` et `Tenure_Days`.
   - Création de features : `Total_Spending`, `Total_Purchases`, `Dependents`, etc.

2. Analyse exploratoire (EDA)
   - Distributions d'âge, ancienneté, dépenses par catégorie, corrélations, taux d'acceptation par campagne.

3. Segmentation (Clustering)
   - Standardisation des variables financières/démographiques.
   - K-Means (ex. k=4) pour obtenir des clusters représentatifs.
   - Profilage des clusters (moyennes par cluster, compte clients).

4. Modélisation prédictive
   - Construction d'un modèle Random Forest (par campagne) pour prédire l'acceptation (1/0).
   - Évaluation via AUC, précision, rappel, F1-score.
   - Utilisation de la feature `Cluster` comme variable explicative.

5. Prédiction et recommandation
   - Pour un client donné (mock ou réel), le pipeline estime la probabilité d'acceptation pour chaque campagne et recommande la campagne avec la probabilité la plus élevée.

## Résultats clés (extrait)

- Le notebook imprime une synthèse AUC / F1 par campagne et affiche un barplot des AUC.
- Les clusters identifient des segments tels que : clients à haute valeur, clients sensibles aux promotions, jeunes valeur faible, etc.

## Comment reproduire (exemples PowerShell)

1) Créer un environnement (optionnel) et installer les dépendances :

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

2) Lancer Jupyter Notebook / Lab et ouvrir `ML3.ipynb` :

```powershell
jupyter notebook
# ou
jupyter lab
```

3) Exécuter les cellules dans l'ordre. Le notebook lit/écrit `Camp_Market_cleaned.csv` : si vous partez du CSV brut, exécutez d'abord la cellule de nettoyage pour générer la version nettoyée.

## Dépendances principales

- Python 3.8+ (teste avec 3.8–3.11)
- pandas, numpy
- matplotlib, seaborn
- scikit-learn

Un `requirements.txt` minimal recommandé :

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

## Prochaines étapes (suggestions)

- Ajouter un `requirements.txt` et/ou `environment.yml` pour rendre l'installation reproductible.
- Ajouter des tests unitaires pour les fonctions de prétraitement et prédiction.
- Évaluer d'autres modèles (XGBoost, Logistic Regression) et calibrer les probabilités.
- Mettre en place une validation croisée plus robuste et un contrôle des déséquilibres (SMOTE, under/over-sampling).
- Déployer un petit service (Flask/FastAPI) pour fournir des recommandations en ligne pour de nouveaux clients.

## Licence & Contact

Ce projet est à usage pédagogique (Epitech). Pour toute question, modifier le README ou demander des améliorations, contactez l'auteur du dépôt.

---
