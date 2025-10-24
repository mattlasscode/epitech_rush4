import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Configurer le style des visualisations
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

# Charger le jeu de données
# Le fichier utilise des points-virgules comme séparateur
df = pd.read_csv('Camp_Market.csv', sep=';')

# --- Nettoyage et prétraitement des données ---

# Supprimer les colonnes non pertinentes
df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

# Remplir les valeurs manquantes de 'Income' par la moyenne
df['Income'] = df['Income'].fillna(df['Income'].mean())

# Convertir 'Dt_Customer' en type datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# --- Création de nouvelles variables ---

# Estimer l'âge du client à partir de l'année de naissance
df['Age'] = 2015 - df['Year_Birth']

# Nombre de jours depuis l'inscription (référence: 01/01/2015)
df['Tenure_Days'] = (pd.to_datetime('2015-01-01') - df['Dt_Customer']).dt.days

# Somme des montants dépensés par catégorie
spending_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
df['Total_Spending'] = df[spending_cols].sum(axis=1)

# Total des achats selon les différents canaux
purchase_cols = [
    'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumDealsPurchases'
]
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

# Nombre total d'enfants à la maison
df['Dependents'] = df['Kidhome'] + df['Teenhome']

# Retirer les âges manifestement erronés (par ex. > 100 ans)
df = df[df['Age'] < 100]

# Encoder les variables catégorielles pour le clustering
df_encoded = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

print("Préparation des données terminée.")
print("Informations sur le DataFrame mises à jour :")
df_encoded.info()