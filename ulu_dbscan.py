import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

data = pd.read_csv("ulu.csv")
# Load your dataset. Replace the following line with loading your actual dataset.
data = data[['x', 'y', 'z']].values


# DBScan clustering
dbscan = DBSCAN()
dbscan_labels = dbscan.fit_predict(data)
n_clusters = len(np.unique(dbscan_labels)) - 1  # Subtract one to exclude noise points
print(f"Number of clusters in DBScan: {n_clusters}")

# Scatter plots for pairs of features with different colors for DBScan clusters
for i in range(data.shape[1]):
    for j in range(i + 1, data.shape[1]):
        plt.scatter(data[:, i], data[:, j], c=dbscan_labels)
        plt.xlabel(f"Feature {i}")
        plt.ylabel(f"Feature {j}")
        plt.title("DBScan Clusters")
        plt.show()

# K-Means clustering
kmeans = KMeans(n_clusters=n_clusters)  # Use the same number of clusters as DBScan
kmeans_labels = kmeans.fit_predict(data)
print("K-Means Labels:", kmeans_labels)


# Scatter plots for pairs of features with different colors for K-Means clusters
for i in range(data.shape[1]):
    for j in range(i + 1, data.shape[1]):
        plt.scatter(data[:, i], data[:, j], c=kmeans_labels)
        plt.xlabel(f"Feature {i}")
        plt.ylabel(f"Feature {j}")
        plt.title("K-Means Clusters")
        plt.show()


# Silhouette Scores
dbscan_silhouette = silhouette_score(data, dbscan_labels)
kmeans_silhouette = silhouette_score(data, kmeans_labels)
print(f"Silhouette Score for DBScan: {dbscan_silhouette}")
print(f"Silhouette Score for K-Means: {kmeans_silhouette}")

# PCA for 3D visualization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_scaled)

# Create a 2D scatter plot of the PCA results with colors representing DBScan clusters
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=dbscan_labels)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA results in 2D, colored by DBScan clusters')
plt.show()

# Create a 3D scatter plot of the PCA results with colors representing DBScan clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca[:, 0], data_pca[:, 1], data_pca[:, 2], c=dbscan_labels)
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA results in 3D, colored by DBScan clusters')
plt.show()