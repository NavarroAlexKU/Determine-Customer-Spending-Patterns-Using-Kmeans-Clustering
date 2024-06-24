# Determine Customer Spending Patterns Using K-means Clustering

![kmeans](https://github.com/NavarroAlexKU/Determine-Customer-Spending-Patterns-Using-Kmeans-Clustering/blob/main/Kmeans%20Clustering.jpg?raw=true)

## Table of Contents
- [Determine Customer Spending Patterns Using K-means Clustering](#determine-customer-spending-patterns-using-k-means-clustering)
  - [Table of Contents](#table-of-contents)
  - [Project Objective](#project-objective)
  - [Importing Python Packages](#importing-python-packages)
  - [Importing the Dataset](#importing-the-dataset)
  - [Using the Elbow Method to Find the Optimal Number of Clusters](#using-the-elbow-method-to-find-the-optimal-number-of-clusters)
  - [Training the Final Model and Visualizing Clusters](#training-the-final-model-and-visualizing-clusters)

## Project Objective
Analyze customer spending habits using K-means clustering on annual income and spending score data. The goal is to identify distinct customer groups for targeted marketing and personalized services. By visualizing the clusters, I aim to gain insights into spending patterns and relationships between income and behavior.

### KMeans Clustering Steps:
1. **Choose the number of K clusters**: Determine the optimal number of clusters (K) using the Elbow method or another heuristic.
2. **Select random K points as centroids**: Initialize K centroids randomly.
3. **Assign each data point to the closest centroid**: Form K clusters by assigning each data point to its nearest centroid.
4. **Compute new centroids**: Calculate the mean of the data points in each cluster to find the new centroid.
5. **Reassign data points**: Reassign each data point to the new closest centroid. Repeat steps 4 and 5 until the centroids no longer change significantly.

## Importing Python Packages
Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.

```python
# Import python packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

print(os.environ['OMP_NUM_THREADS'])
```
# Importing the Dataset:
```
df = pd.read_csv(path_to_dataset)
```

## Using the Elbow Method to Find the Optimal Number of Clusters
- Create a 2-dimensional array of the independent variables.
- Compute WCSS for different numbers of clusters.
- Plot the results to identify the optimal number of clusters.
```
# Create 2 dimensional array of our independent variables:
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Create empty list to store wcss values:
wcss = []

# Loop through X array and compute optimal number of clusters using the WCSS algorithm:
for i in range(1, 11):
    # Instantiate kmeans:
    kmeans = KMeans(
        # set number of clusters:
        n_clusters = i,
        # set init:
        init = "k-means++",
        # set random state:
        random_state = 42,
    )
    # Train model:
    kmeans.fit(X)
    # Append the wcss per cluster to wcss list:
    wcss.append(kmeans.inertia_)

# Plot the results:
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel('Number of Clusters')
plt.ylabel("WCSS")
plt.tight_layout()
plt.show()
```

## Training the Final Model and Visualizing Clusters
- Instantiate the K-means model with the optimal number of clusters.
- Fit the model and predict the clusters.
- Print the cluster centers and labels.
- Visualize the clusters and centroids.
```
# Instantiate kmeans:
kmeans = KMeans(
    # set number of clusters using the optimal number of clusters found above:
    n_clusters = 5,
    # set init:
    init = "k-means++",
    # set random state:
    random_state = 42,
)

# Fit model:
y_pred = kmeans.fit_predict(X)

# Print the cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Print the labels of the clusters
print("Cluster labels for each data point:")
print(kmeans.labels_)

# Plot the clusters
plt.figure(figsize=(10, 6))

# Colors for different clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(X[y_pred == i, 0], X[y_pred == i, 1], s=100, c=colors[i], label=f'Cluster {i}')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', edgecolor='black')

# Add titles and labels
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.tight_layout()
plt.show()
```
