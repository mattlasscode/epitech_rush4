import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

df = pd.read_csv('Camp_Market.csv', sep=';')

df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

df['Income'] = df['Income'].fillna(df['Income'].mean())

df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

df['Age'] = 2015 - df['Year_Birth']

df['Tenure_Days'] = (pd.to_datetime('2015-01-01') - df['Dt_Customer']).dt.days

spending_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
df['Total_Spending'] = df[spending_cols].sum(axis=1)

purchase_cols = [
    'NumWebPurchases', 'NumCatalogPurchases', 
    'NumStorePurchases', 'NumDealsPurchases'
]
df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

df['Dependents'] = df['Kidhome'] + df['Teenhome']

df = df[df['Age'] < 100]

df_encoded = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

print("Préparation des données terminée.")
print("Informations mises à jour :")
df_encoded.info()