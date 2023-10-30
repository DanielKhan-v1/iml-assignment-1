import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, silhouette_samples

# Load the Iris dataset
df = pd.read_csv("iris.csv")

# Remove the non-numeric 'species' column
df_numeric = df.drop('species', axis=1)

# Choose the number of clusters for k-Means
num_clusters = 3

# Initialize an empty list to store the sum of squared distances
ssd = []

# Set a fixed random seed for K-Means
random_seed = 42

# Try k from 1 to 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=random_seed)
    kmeans.fit(df_numeric)  # Fit K-Means to the numeric data
    ssd.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(range(1, 11), ssd, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Create a K-Means model with the optimal number of clusters (k=3)
kmeans = KMeans(n_clusters=3, random_state=random_seed)

# Fit the K-Means model to the numeric data
kmeans.fit(df_numeric)

# Add the cluster labels to your original DataFrame
df['cluster'] = kmeans.labels_

# Convert string labels to integer labels
label_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
df['true_label'] = df['species'].map(label_mapping)

# Calculate accuracy and confusion matrix
true_labels = df['true_label']
accuracy = accuracy_score(true_labels, df['cluster'])
confusion = confusion_matrix(true_labels, df['cluster'])

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

# Visualize the results with a pairplot, coloring by cluster
g = sns.pairplot(df, hue="cluster", height=1.5, aspect=1.5)
plt.show()

# Create scatter plots for pairwise combinations of columns within each cluster
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    sns.pairplot(cluster_data, height=1.25, aspect=1.25) 
    
    plt.suptitle(f'Pairwise Scatter Plots for Cluster {cluster}')
    plt.show()

# Compute silhouette score for the clustering
silhouette_avg = silhouette_score(df_numeric, df['cluster'])
print("Average Silhouette Score:", silhouette_avg)

# Compute and report silhouette scores for each individual data point
sample_silhouette_values = silhouette_samples(df_numeric, df['cluster'])
df['silhouette_score'] = sample_silhouette_values
print("Individual Silhouette Scores:")
print(df[['species', 'cluster', 'silhouette_score']])
# Load the unknown dataset and remove the 'id' column
unknown_df = pd.read_csv("unknown_species.csv")
unknown_numeric = unknown_df.drop('id', axis=1)

print(unknown_df)
# Use the trained K-Means model to predict the clusters for the unknown flowers
unknown_clusters = kmeans.predict(unknown_df)

# Map cluster labels to species, including 'unknown'
cluster_to_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica', -1: 'unkown'}
unknown_species = [cluster_to_species[cluster] for cluster in unknown_clusters]

print("Predicted Species for Unknown Flowers:")
print(unknown_species)
