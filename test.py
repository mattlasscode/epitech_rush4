import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Set up visualization styles
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-whitegrid')

# Load the dataset
# The dataset is delimited by semicolons, so we specify 'sep=';'
df = pd.read_csv('Camp_Market.csv', sep=';')

# --- Data Cleaning and Preprocessing ---

# Drop irrelevant columns as specified in the prompt
df = df.drop(columns=['Z_CostContact', 'Z_Revenue'])

# Fill missing values in 'Income' with the mean income
df['Income'] = df['Income'].fillna(df['Income'].mean())

# Convert 'Dt_Customer' to datetime objects
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])

# --- Feature Engineering ---

# Calculate customer's age from birth year
df['Age'] = 2015 - df['Year_Birth']

# Calculate the number of days since the customer's enrollment with the company
df['Tenure_Days'] = (pd.to_datetime('2015-01-01') - df['Dt_Customer']).dt.days

# Calculate total spending and total purchases
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

# Calculate total number of children at home
df['Dependents'] = df['Kidhome'] + df['Teenhome']

# Remove rows with absurd age (e.g., Year_Birth before 1900)
df = df[df['Age'] < 100]

# Create dummy variables for categorical features for clustering
df_encoded = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

print("Data Preparation Complete.")
print("Updated DataFrame Info:")
df_encoded.info()