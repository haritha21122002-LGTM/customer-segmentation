
# ============================================================
# STORE CUSTOMER SEGMENTATION
# Unsupervised Learning - K-Means Clustering
# Dataset: store_customers.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("store_customers.csv")

print("\n========== DATASET INFORMATION ==========")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDataset Information:")
print(df.info())

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

df = df.drop_duplicates()

features = [
    'Age',
    'Annual Income (k$)',
    'Spending Score (1-100)'
]

# Convert selected columns into numeric values
df[features] = df[features].apply(
    pd.to_numeric, errors='coerce'
)

# Fill missing values
df[features] = df[features].fillna(
    df[features].median()
)

print("\nCleaned Dataset Shape:", df.shape)

# ------------------------------------------------------------
# 3. SELECT FEATURES
# ------------------------------------------------------------

X = df[features]

# ------------------------------------------------------------
# 4. FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature Scaling Completed!")

# ------------------------------------------------------------
# 5. ELBOW METHOD
# ------------------------------------------------------------

inertia = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(9, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker='o',
    color='#536DFE',
    linewidth=2
)

plt.title("Elbow Method - Optimal Clusters",
          fontsize=15, fontweight='bold')

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.grid(alpha=0.3)
plt.show()

# ------------------------------------------------------------
# 6. K-MEANS CLUSTERING
# ------------------------------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n========== CLUSTER RESULTS ==========")

print(df[['CustomerID', 'Age',
          'Annual Income (k$)',
          'Spending Score (1-100)',
          'Cluster']].head(10))

# ------------------------------------------------------------
# 7. SILHOUETTE SCORE
# ------------------------------------------------------------

score = silhouette_score(
    X_scaled,
    df['Cluster']
)

print("\nSilhouette Score:", round(score, 4))

# ------------------------------------------------------------
# 8. CLUSTER COUNT
# ------------------------------------------------------------

print("\n========== CUSTOMER COUNT ==========")

print(df['Cluster'].value_counts().sort_index())

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x='Cluster',
    hue='Cluster',
    palette='viridis',
    legend=False
)

plt.title("Number of Customers in Each Cluster",
          fontsize=14, fontweight='bold')

plt.xlabel("Customer Cluster")
plt.ylabel("Number of Customers")
plt.show()

# ------------------------------------------------------------
# 9. CUSTOMER SEGMENTATION VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    palette='viridis',
    s=80
)

plt.title("Customer Segmentation using K-Means",
          fontsize=15, fontweight='bold')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Customer Cluster")
plt.grid(alpha=0.2)
plt.show()

# ------------------------------------------------------------
# 10. AGE VS ANNUAL INCOME
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Age',
    y='Annual Income (k$)',
    hue='Cluster',
    palette='viridis',
    s=80
)

plt.title("Age vs Annual Income by Cluster",
          fontsize=15, fontweight='bold')

plt.xlabel("Customer Age")
plt.ylabel("Annual Income (k$)")
plt.show()

# ------------------------------------------------------------
# 11. CLUSTER ANALYSIS
# ------------------------------------------------------------

cluster_analysis = df.groupby('Cluster')[features].mean()

print("\n========== CLUSTER ANALYSIS ==========")
print(cluster_analysis.round(2))

# ------------------------------------------------------------
# 12. CUSTOMER SEGMENT DESCRIPTION
# ------------------------------------------------------------

print("\n========== CUSTOMER SEGMENTS ==========")

for cluster in sorted(df['Cluster'].unique()):

    data = cluster_analysis.loc[cluster]

    print(f"\nCluster {cluster}")

    print("Average Age:",
          round(data['Age'], 2))

    print("Average Annual Income:",
          round(data['Annual Income (k$)'], 2))

    print("Average Spending Score:",
          round(data['Spending Score (1-100)'], 2))

# ------------------------------------------------------------
# 13. SAVE CLUSTERED DATASET
# ------------------------------------------------------------

df.to_csv(
    "customer_segmentation_results.csv",
    index=False
)

print("\n======================================")
print("CUSTOMER SEGMENTATION COMPLETED!")
print("Results saved successfully!")
print("======================================")
