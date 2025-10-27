# Camp_Market — aperçu convivial

Bienvenue ! Ce dépôt rassemble une petite analyse des campagnes marketing d'une enseigne fictive (jeu de données "Camp_Market").

Si vous n'avez pas envie d'entrer dans les détails techniques, voici l'essentiel :

- On a nettoyé les données et créé quelques indicateurs simples (âge, dépenses totales, ancienneté, etc.).
- On a regardé qui dépense quoi et comment les clients réagissent aux anciennes campagnes.
- On a groupé les clients en profils (quelques segments types : gros dépensiers, familles, jeunes clients, ...).
- On a essayé des modèles rapides pour estimer la probabilité qu'un client accepte une offre — puis on peut recommander la campagne la plus prometteuse.

Pourquoi ce repo peut vous intéresser

- Pour comprendre, sans trop de jargon, comment on peut passer de données brutes à des recommandations marketing.
- Pour voir des exemples concrets de nettoyage, visualisations et segmentation.
- Pour jouer avec un notebook et tester des prédictions sur des profils clients fictifs.

Fichiers importants

- `ML3.ipynb` — le notebook principal avec tout : nettoyage, visualisations, clustering et modèles.
- `Camp_Market.csv` — les données brutes.
- `Camp_Market_cleaned.csv` — version nettoyée générée par le notebook.

Comment lancer rapidement

1. (Optionnel) créez un environnement Python et installez les dépendances.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

2. Ouvrez le notebook :

```powershell
jupyter notebook
```

3. Lancez les cellules du notebook dans l'ordre. Si vous partez du CSV brut, exécutez la cellule qui nettoie les données pour obtenir `Camp_Market_cleaned.csv`.

Quelques idées si vous voulez creuser

- Tester d'autres modèles (par exemple XGBoost) et comparer les résultats.
- Améliorer la gestion des classes déséquilibrées (si nécessaire).
- Transformer ce notebook en petit service qui, donné un profil client, renvoie la campagne recommandée.

Besoin d'aide ?

Si vous voulez que je :
- rende le README encore plus court, ou
- ajoute un `requirements.txt`, ou
- crée un script simple pour prédire la meilleure campagne en ligne de commande —

dites-moi lequel et je m'en occupe.

---

Version conviviale du README — reformulée pour être plus accessible et moins technique.
