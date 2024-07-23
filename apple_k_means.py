import numpy as np
from sklearn.cluster import KMeans

# Define the dataset
data = np.array([
[-0.154, 0.376, 0.099],
[-0.103, 0.476, -0.027],
[0.228, 0.036, -0.251],
[0.330, 0.013, -0.251],
[-0.114, 0.482, 0.014],
[0.295, 0.084, -0.297],
[0.262, 0.042, -0.304],
[-0.051, 0.416, -0.306]
])
# Apply k-means clustering with k=2
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

# Get the cluster labels for each point
labels = kmeans.labels_

# Get the coordinates of the cluster centroids
centroids = kmeans.cluster_centers_

# Print the results
print("Cluster Labels:")
print(labels)
print("\nCluster Centroids:")
print(centroids)