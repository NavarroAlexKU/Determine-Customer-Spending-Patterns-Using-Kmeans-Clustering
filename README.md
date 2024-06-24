## Table of Contents
- [Determine Customer Spending Patterns Using K-means Clustering](#determine-customer-spending-patterns-using-k-means-clustering)
  - [Table of Contents](#table-of-contents)
  - [Project Objective](#project-objective)
  - [Importing Python Packages](#importing-python-packages)
  - [Importing the Dataset](#importing-the-dataset)
  - [Data Exploration](#data-exploration)
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
- Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.

## Importing the Dataset
- Load the dataset into a pandas DataFrame.

## Data Exploration
- Display the first few rows of the dataset to understand its structure.

## Using the Elbow Method to Find the Optimal Number of Clusters
- Create a 2-dimensional array of the independent variables.
- Compute WCSS for different numbers of clusters.
- Plot the results to identify the optimal number of clusters.

## Training the Final Model and Visualizing Clusters
- Instantiate the K-means model with the optimal number of clusters.
- Fit the model and predict the clusters.
- Print the cluster centers and labels.
- Visualize the clusters and centroids.
