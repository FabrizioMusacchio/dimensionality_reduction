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
# %% FACTOR ANALYSIS

"""
First check the correlation plot of all the  variables: if any variables are too correlated
with others, considering to remove one of the corresponding columns from the dataset:
"""
sns.heatmap(df.corr())

"""
Adequacy Test
source: https://www.datacamp.com/tutorial/introduction-factor-analysis

Before we perform factor analysis, we need to evaluate the “factorability” 
of our dataset. Factorability means "can we found the factors in the dataset?". 
There are two methods to check the factorability or sampling adequacy:

* Bartlett's Test
* Kaiser-Meyer-Olkin Test

Bartlett's test of sphericity checks whether or not the observed variables 
intercorrelate at all using the observed correlation matrix against the identity 
matrix. If the test found statistically insignificant, you should not employ a factor analysis.
"""
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)
"""
The second values is the p-value of the Bartlett's test. Here: =0.0. Hence, the test was 
statistically significant, indicating that the observed correlation matrix is not an 
identity matrix.
"""

"""
Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. 
It determines the adequacy for each observed variable and for the complete model. 
KMO estimates the proportion of variance among all the observed variable. 
Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1. 
Value of KMO less than 0.6 is considered inadequate.
"""
kmo_all,kmo_model=calculate_kmo(df)
print(kmo_model)

"""
For choosing the number of factors, we can use a scree plot based on Eigenvalues. 
An eigenvalue of more than one means that the factor explains more variance than a 
unique variable. An eigenvalue of 2.5 means that the factor would explain the variance 
of 2.5 variables, and so on. Here, we choose the number of Eigenvalues higher than 1 
to be considered as the number of factors:
"""
# Perform preliminary factor analysis:
fa = FactorAnalyzer()
fa.fit(df)

# Check Eigenvalues:
ev, v = fa.get_eigenvalues()
print("Eigenvalues:", ev)

# Scree plot for validating the number of factors:
plt.figure(figsize=(5, 3))
plt.scatter(range(1, df.shape[1]+1), ev)
plt.plot(range(1, df.shape[1]+1), ev)
plt.title('Scree Plot')
plt.xlabel('Faktor')
plt.ylabel('Eigenwert')
plt.grid()
plt.show()

"""
In our example, we can see that only the first three Eigenvalues are higher than 1. 
Thus, three factors should be chosen:
"""
# Re-perform factor analysis, now with 2 factors:
fa = FactorAnalyzer(n_factors=3, rotation='varimax')
fa.fit(df)


"""
The factor loading is a matrix which shows the relationship of each variable to the 
underlying factor. It shows the correlation coefficient for the observed variable and 
factor. It shows the variance explained by the observed variables (source).
The higher a factor loading, the more important a variable is for said factor. 

https://www.datacamp.com/tutorial/introduction-factor-analysis
"""
# Extract factor loadings and factors:
factor_loadings = fa.loadings_
factor_communalities = fa.get_communalities()
factors = fa.transform(df)

"""
'factors' is the actual dimensionally reduced dataset, i.e. our "new" dataset consisting of only
two (new) dependent variables, that describe the underlying classes of the iris dataset well.
"""

# Create a dataframe for the factor loadings:
df_loadings = pd.DataFrame(data=factor_loadings, columns=['Factor 1', 'Factor 2', 'Factor 3'])
df_loadings['Feature'] = input_data.feature_names

# Visualize the factor loadings:
plt.bar(df_loadings['Feature'], df_loadings['Factor 1'], label='Factor 1')
plt.bar(df_loadings['Feature'], df_loadings['Factor 2'], label='Factor 2')
plt.bar(df_loadings['Feature'], df_loadings['Factor 3'], label='Factor 3')
plt.xlabel('Feature')
plt.ylabel('Factor Loading')
plt.title('Factor Loadings of Iris Dataset')
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

Z=np.abs(factor_loadings)
plt.pcolor(Z)
plt.colorbar()
ax = plt.gca()
ax.set_yticks(np.arange(factor_loadings.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(factor_loadings.shape[1])+0.5, minor=False)
ax.set_yticklabels(df.keys())
ax.set_xticklabels(["Factor 1", "Factor 2", "Factor 3"])
plt.tight_layout()
plt.savefig('factor_loadings.png', dpi=300)
plt.show()


# Get variance of each factors
factor_variance = fa.get_factor_variance()
factor_variance_df = pd.DataFrame(data=factor_variance, columns=['Factor 1', 'Factor 2','Factor 2'], 
                                  index=['SS Loadings', 'Proportion Var', 'Cumulative Var'])
print(factor_variance_df)
"""
XY% of the cumulative variance in the data can be explained by the two factors.
"""

"""
How do we know if our factors are any good? Use the Cronbach alpha 
to measure whether or not the variables of a factor  form a “coherent” and 
reliable factor. A value above 0.6 for the alpha is in  practice deemed acceptable:
"""

""" 
factor1 = df[['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']]
factor2 = df[['sepal width (cm)']]
factor1_alpha = pg.cronbach_alpha(factor1)
#factor2_alpha = pg.cronbach_alpha(factor2) # works only, when factor2 consists of more than one variable  """

# visualize the factors together with the original classes in order to see if the factors
# separate the classes well:
colors = ['r', 'g', 'b']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    ax.scatter(factors[indices, 0], factors[indices, 1], factors[indices, 2], c=color, label=target)
ax.view_init(elev=35, azim=45)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('Factors of Wine Dataset')
ax.zaxis.labelpad=-3.9
ax.legend()
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Adjust the figure size as needed
# "XY" projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[0].scatter(factors[indices, 0], factors[indices, 1], c=color, label=target)
axs[0].set_xlabel('Factor 1')
axs[0].set_ylabel('Factor 2')
axs[0].set_title('XY Projection')
# "XZ" projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[1].scatter(factors[indices, 0], factors[indices, 2], c=color, label=target)
axs[1].set_xlabel('Factor 1')
axs[1].set_ylabel('Factor 3')
axs[1].set_title('XZ Projection')
# "YZ" projection:
for target, color in zip(target_names, colors):
    indices = input_data.target == input_data.target_names.tolist().index(target)
    axs[2].scatter(factors[indices, 1], factors[indices, 2], c=color, label=target)
axs[2].set_xlabel('Factor 2')
axs[2].set_ylabel('Factor 3')
axs[2].set_title('YZ Projection')
plt.show()


""" Verify reproducibility by "blindly" (unsupervised) 
identifying clusters from the factors (i.e., do the 
clusters match with the actual classes?): """
dbscan = DBSCAN(eps=0.46, min_samples=5)
dbscan_labels = dbscan.fit_predict(factors)
dbscan_labels_unique = set(dbscan_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = factors[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = factors[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
ax.view_init(elev=35, azim=45)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('DBSCAN Clustering of the Factors')
ax.zaxis.labelpad=-3.9
ax.legend()
plt.show()
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Adjust the figure size as needed
# XY projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = factors[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = factors[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"Cluster {label+1}")
axs[0].set_xlabel('Factor 1')
axs[0].set_ylabel('Factor 2')
axs[0].set_title('DBSCAN classes XY Projection')
# XZ projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = factors[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = factors[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[1].set_xlabel('Factor 1')
axs[1].set_ylabel('Factor 3')
axs[1].set_title('DBSCAN classes XZ Projection')
# YZ projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = factors[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = factors[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[2].set_xlabel('Factor 2')
axs[2].set_ylabel('Factor 3')
axs[2].set_title('DBSCAN classes YZ Projection')



kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(factors)
kmeans_labels_unique = set(kmeans_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in kmeans_labels_unique:
    cluster_points = factors[kmeans_labels==label]
    ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
ax.view_init(elev=35, azim=45)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('KMEANS Clustering of the Factors')
ax.zaxis.labelpad=-3.9
ax.legend()
plt.show()
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  # Adjust the figure size as needed
# XY projection:
for label in kmeans_labels_unique:
    cluster_points = factors[kmeans_labels==label]
    axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"Cluster {label+1}")
axs[0].set_xlabel('Factor 1')
axs[0].set_ylabel('Factor 2')
axs[0].set_title('KMEANS classes XY Projection')
# XZ projection:
for label in kmeans_labels_unique:
    cluster_points = factors[kmeans_labels==label]
    axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[1].set_xlabel('Factor 1')
axs[1].set_ylabel('Factor 13')
axs[1].set_title('KMEANS classes XZ Projection')
# YZ projection:
for label in kmeans_labels_unique:
    cluster_points = factors[kmeans_labels==label]
    axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[2].set_xlabel('Factor 2')
axs[2].set_ylabel('Factor 3')
axs[2].set_title('KMEANS classes YZ Projection')
# %% END