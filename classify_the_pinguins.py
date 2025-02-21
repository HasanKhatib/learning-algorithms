"""
Lab 3: 3.4 Classify the Penguins
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.stats import mode

# Step 1: Load & Preprocess Data
print("Loading dataset...")
df = pd.read_csv("data/penguins.csv")

# Select relevant columns
df = df[["species", "bill_length_mm", "bill_depth_mm"]]

# Drop missing values
df.dropna(inplace=True)

# Extract features for clustering
X_penguins = df[["bill_length_mm", "bill_depth_mm"]]

# Encode species labels for accuracy calculation
species_encoder = LabelEncoder()
df["species_encoded"] = species_encoder.fit_transform(df["species"])

print("Dataset loaded and preprocessed.\n")

# Step 2: Apply K-Means Clustering
print("Applying K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_penguins)

# Get cluster centroids
centroids = kmeans.cluster_centers_

print("K-Means Clustering Completed.\n")

# Step 3: Visualization of Clusters
print("Plotting cluster visualization...")

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue=df["cluster"], palette="viridis", alpha=0.7)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")

plt.title("K-Means Clustering of Penguins")
plt.legend()
plt.show()

# Step 4: Evaluate Accuracy
print("Evaluating clustering accuracy...")

# Function to map cluster labels to actual species labels
def map_clusters_to_species(true_labels, cluster_labels):
    mapped_labels = np.zeros_like(cluster_labels)
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        mapped_labels[mask] = mode(true_labels[mask])[0]
    return mapped_labels

# Remap clusters to best-matching species
mapped_clusters = map_clusters_to_species(df["species_encoded"].values, df["cluster"].values)

# Compute accuracy
kmeans_accuracy = accuracy_score(df["species_encoded"], mapped_clusters)
print(f"K-Means Clustering Accuracy: {kmeans_accuracy:.2f}")
