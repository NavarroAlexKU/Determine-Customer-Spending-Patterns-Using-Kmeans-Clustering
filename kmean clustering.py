# %% [markdown]
# ### Import Python Packages:

# %%
# Import python packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

print(os.environ['OMP_NUM_THREADS'])
# Import dataset:
df = pd.read_csv('\Clustering\data.csv')

# %%
# Check first 5 rows of data:
df.head()

# %% [markdown]
# Use elbow method to find the optimal number of clusters:

# %%
# Create 2 dimensional array of our independent variables:
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# %%
# create empty list to store wcss values:
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

# %% [markdown]
# ### Train Final Model & Visualize Clusters:

# %%
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

# %%


# %%


# %%


# %%



