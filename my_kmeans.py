import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score


# ========================================================
# C.I. Iris clustering with K-means
# Exercise 6

# Function to implement K-Means clustering
def kmeans(x, k, no_of_iterations):
    # Randomly initialize centroids by selecting 'k' data points
    idx = np.random.choice(len(x), k, replace=False)
    centroids = x[idx, :]

    # Calculate the Euclidean distances between data points and centroids
    distances = cdist(x, centroids, 'euclidean')

    # Assign each data point to the cluster with the nearest centroid
    points = np.array([np.argmin(i) for i in distances])

    # Iterate to update centroids and reassign data points to clusters
    for _ in range(no_of_iterations):
        centroids = []
        for idx in range(k):
            # Update centroids by computing the mean of data points in the cluster
            temp_cent = x[points == idx].mean(axis=0)
            centroids.append(temp_cent)

        centroids = np.vstack(centroids)  # Updated Centroids

        # Recalculate distances and reassign data points to the nearest centroids
        distances = cdist(x, centroids, 'euclidean')
        points = np.array([np.argmin(i) for i in distances])

    return points


if __name__ == '__main__':
    # Load the Iris dataset
    iris_df = pd.read_csv('iris.csv')

    # Encode the 'species' column
    label_encoder = LabelEncoder()
    iris_df['species'] = label_encoder.fit_transform(iris_df['species'])

    # Separate features and labels
    X = iris_df.drop('species', axis=1)
    y = iris_df['species']
    n_samples = len(X)  # Number of samples in the dataset

    # Run the custom k-means implementation
    k = 3  # Number of clusters
    labels = kmeans(X.values, k, 1000)

    # Assign cluster labels to all samples
    all_labels = np.repeat(labels, n_samples)

    # Compute the silhouette score for the clustering result
    silhouette_avg = silhouette_score(X, labels)
    print(f"Silhouette Score of k-means implementation: {silhouette_avg}")

    for cluster in range(k):
        # Plot data points for each cluster
        plt.scatter(X[labels == cluster].iloc[:, 0], X[labels == cluster].iloc[:, 1], label=f'Cluster {cluster}')

    # Plot the centroids (cluster centers) as 'x' markers in red
    centroids = np.array([X[labels == cluster].mean(axis=0) for cluster in range(k)])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')

    # Set plot labels and title, and display the legend
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('My K-Means Clustering')
    plt.legend()

    # Show the plot
    plt.show()
