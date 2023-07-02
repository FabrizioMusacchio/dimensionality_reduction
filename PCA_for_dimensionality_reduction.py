"""
A script to compare PCA, FA and AE on a dataset for dimensionality reduction.

author: Fabrizio Musacchio (fabriziomusacchio.com)
date: June 19, 2023


For reproducibility:

conda create -n vscode_testruns python=3.10
conda activate vscode_testruns
conda install -y matplotlib numpy pandas scikit-learn seaborn ipykernel scikit-image
conda install -c desilinguist factor_analyzer

on linux/windows:
conda install -y tensorflow

on a mac:
conda install -c apple tensorflow-deps
conda install pip
pip install tensorflow-macos tensorflow-metal tensorflow_datasets
"""
# %% IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.optimizers import Adam
import random
import pingouin as pg
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
# %% LOADING THE DATASET AND TAKE A FIRST LOOK AT IT
input_data = load_wine()
X = input_data.data

feature_names = input_data.feature_names

# Many dimensionality reduction algorithms as affected by scale, so we need to scale
# the features in our data before applying. e.g., PCA or FA. We can use StandardScaler 
# to standardize the data set’s features onto unit scale (mean = 0 and variance = 1).
# If we don’t scale our data, it can have a negative effect on our algorithm:
X_Factor = StandardScaler().fit_transform(X)
df = pd.DataFrame(data=X_Factor, columns=input_data.feature_names)

df_full = df.copy()
df_full['Target'] = input_data.target
target_names = input_data.target_names

# let's explore the data first with a scatter plot of the features:
sns.set(style="ticks")
sns.pairplot(df_full, hue='Target', markers=['o', 's', 'D'], palette="viridis") # viridis, Set1
plt.suptitle('Pairwise relationships in the (high-dimensional) Wine dataset.', y=1.02)
plt.show()
# %% PCA

# Explore the number of necessary PCA:
pca_explore = PCA(n_components=13)
pcas_explore = pca_explore.fit_transform(X_Factor)
ev = pca_explore.singular_values_

# Scree plot for validating the number of factors:
plt.figure(figsize=(5, 3))
plt.scatter(range(1, df.shape[1]+1), ev)
plt.plot(range(1, df.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('PC#')
plt.ylabel('Singular Value')
plt.ylim(0, 30)
plt.grid()
plt.show() 


"""
The explained variance tells us how much information (variance) can be attributed 
to each of the principal components. This is important because while we can convert 
high-dimensional space to a two- or three-dimensional space, we lose some of the variance 
(information). By using the attribute explained_variance_ratio, we can see that the 
first principal component contains XY percent of the variance,  the second XY percent and
the third XY percent of the variance. Together, the three components contain XY percent of the 
information.

btw., variance_explained_ratio = eigenvalues / np.sum(eigenvalues)
"""
var=np.cumsum(np.round(pca_explore.explained_variance_ratio_, decimals=3) *100)
plt.plot(var)
plt.ylabel("% Variance Explained")
plt.xlabel("# of PCs") 
plt.title ("PCA Variance Explained")
plt.ylim(min(var), 100.5) 
#plt.style.context ('seaborn-whitegrid') 
plt.axhline(y=80, color='r', linestyle='--')
plt.show()

print(f"Explained variances for all 13 PCs:\n {pca_explore.explained_variance_ratio_}\n")
print(f"Cumulative explained variance for the first 3 PCs: {np.sum(pca_explore.explained_variance_ratio_[0:3])}")



# perform PCA with 3 components:
pca = PCA(n_components=3)
pcas = pca.fit_transform(X_Factor)

# create a dataframe with the 3 components and the target variable:
principal_df = pd.DataFrame(data=pcas, columns=['PC1', 'PC2', 'PC3'])
final_df = pd.concat([principal_df, pd.DataFrame(data=input_data.target, columns=['target'])], axis=1)

# visualize the 3 components:
colors = ['r', 'g', 'b']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    ax.scatter(pcas[indices, 0], pcas[indices, 1], pcas[indices, 2], c=color, label=target)
ax.view_init(elev=35, azim=45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad=-5.5
ax.set_title('PCA of Wine Dataset')
ax.legend()
plt.show()

# projections:
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
# XY projection: 
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[0].scatter(pcas[indices, 0], pcas[indices, 1], c=color, label=target)
axs[0].set_xlabel('PCA 1')
axs[0].set_ylabel('PCA 2')
axs[0].set_title('XY Projection')

# XZ projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[1].scatter(pcas[indices, 0], pcas[indices, 2], c=color, label=target)
axs[1].set_xlabel('PCA 1')
axs[1].set_ylabel('PCA 3')
axs[1].set_title('XZ Projection')

# YZ projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[2].scatter(pcas[indices, 1], pcas[indices, 2], c=color, label=target)
axs[2].set_xlabel('PCA 2')
axs[2].set_ylabel('PCA 3')
axs[2].set_title('YZ Projection')
plt.show()


# Verify reproducibility by "blindly" identifying c­lusters from the factors (i.e., 
# do the blind c­lusters match with the actual classes?): 
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(pcas)
dbscan_labels_unique = set(dbscan_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points =  pcas[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points =  pcas[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
ax.view_init(elev=35, azim=45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad=-5.5
ax.set_title('DBSCAN Clustering of the PC')
ax.legend()
plt.show()
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4)) 
# XY projection: [h33]
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = pcas[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = pcas[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"[Cluster] {label+1}")
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('DBSCAN classes XY Projection')
# XZ projection: [h34]
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = pcas[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = pcas[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"[Cluster] {label+1}")
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC3')
axs[1].set_title('DBSCAN classes XZ Projection')
# YZ projection: [h35]
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = pcas[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = pcas[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"[Cluster] {label+1}")
axs[2].set_xlabel('PC2')
axs[2].set_ylabel('PC3')
axs[2].set_title('DBSCAN classes YZ Projection')


kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(pcas)
kmeans_labels_unique = set(kmeans_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in kmeans_labels_unique:
    cluster_points = pcas[kmeans_labels==label]
    ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
ax.view_init(elev=35, azim=45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3', rotation=90)
ax.zaxis.labelpad=-5.5
ax.set_title('KMEANS Clustering of the PC')
ax.legend()
plt.show()
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
# XY projection: 
for label in kmeans_labels_unique:
    cluster_points = pcas[kmeans_labels==label]
    axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"Cluster {label+1}")
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title('KMEANS classes XY Projection')
# XZ projection: 
for label in kmeans_labels_unique:
    cluster_points = pcas[kmeans_labels==label]
    axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC13')
axs[1].set_title('KMEANS classes XZ Projection')
# YZ projection: 
for label in kmeans_labels_unique:
    cluster_points = pcas[kmeans_labels==label]
    axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[2].set_xlabel('PC2')
axs[2].set_ylabel('PC3')
axs[2].set_title('KMEANS classes YZ Projection')
# %% END