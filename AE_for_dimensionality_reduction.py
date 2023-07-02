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
# %% AUTOENCODER

# for reproducibility:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

# number of features:
num_features = X_Factor.shape[1]

# define the dimensions of the encoded and decoded spaces:
encoding_dim = 3

# input layer:
input_layer = Input(shape=(num_features,))

# coding layer:
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# decoding layer:
decoded = Dense(num_features, activation='sigmoid')(encoded)

# create the autoencoder model:
autoencoder = Model(inputs=input_layer, outputs=decoded)

# compile and train the autoencoder model:
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_Factor, X_Factor, epochs=200, batch_size=16, shuffle=True, verbose=0)

# create a separate encoder model:
encoder = Model(inputs=input_layer, outputs=encoded)

# apply the encoder to the input data:
encoded_data = encoder.predict(X_Factor)

# create a dataframe with the encoded data:
df_encoded = pd.DataFrame(data=encoded_data, columns=['Feature 1', 'Feature 2', 'Feature 3'])
df_encoded['Target'] = input_data.target


# visualization of the encoded data:
colors = ['r', 'g', 'b']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for target, color in zip(target_names, colors):
    indices = df_encoded['Target'] == input_data.target_names.tolist().index(target)
    ax.scatter(df_encoded.loc[indices, 'Feature 1'], 
               df_encoded.loc[indices, 'Feature 2'],
               df_encoded.loc[indices, 'Feature 3'], c=color, label=target)
ax.view_init(elev=35, azim=45)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3', rotation=90)
ax.zaxis.labelpad=-4.9
ax.set_title('Autoencoder Encoding of Wine Dataset')
ax.legend()
plt.show()

# projections
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# XY projection: 
for target, color in zip(target_names, colors):
    indices = df_encoded['Target'] == input_data.target_names.tolist().index(target)
    axs[0].scatter(df_encoded.loc[indices, 'Feature 1'], df_encoded.loc[indices, 'Feature 2'], c=color, label=target)
axs[0].set_xlabel('Feature 1')
axs[0].set_ylabel('Feature 2')
axs[0].set_title('XY Projection')

# XZ projection:
for target, color in zip(target_names, colors):
    indices = df_encoded['Target'] == input_data.target_names.tolist().index(target)
    axs[1].scatter(df_encoded.loc[indices, 'Feature 1'], df_encoded.loc[indices, 'Feature 3'], c=color, label=target)
axs[1].set_xlabel('Feature 1')
axs[1].set_ylabel('Feature 3')
axs[1].set_title('XZ Projection')

# YZ projection:
for target, color in zip(target_names, colors):
    indices = df_encoded['Target'] == input_data.target_names.tolist().index(target)
    axs[2].scatter(df_encoded.loc[indices, 'Feature 2'], df_encoded.loc[indices, 'Feature 3'], c=color, label=target)
axs[2].set_xlabel('Feature 2')
axs[2].set_ylabel('Feature 3')
axs[2].set_title('YZ Projection')
plt.show()
# %% AE INTERPRETATION

# Extracting the weights of the neurons in the hidden layer.
"""
The weights indicate how much each feature contributes to building the coded 
Representation of the data in the Wine dataset.
"""
hidden_layer_weights = autoencoder.layers[1].get_weights()[0]

# create a DataFrame for the weights:
df_weights = pd.DataFrame(data=hidden_layer_weights, columns=['Factor 1', 'Factor 2', 'Factor 3'], 
                          index=feature_names)

# visualization of the weights:
fig, axs = plt.subplots(nrows=int(np.ceil(num_features/3)), ncols=3, figsize=(9, 12))
for i, feature in enumerate(feature_names):
    ax = axs[i//3, i%3]#axs[i]
    weights = df_weights.loc[feature].values
    factors = ['Factor 1', 'Factor 2', 'Factor 3']
    colors = ['grey' if w < 0 else 'blue' for w in weights]
    ax.bar(factors, weights, color=colors)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Weight')
    ax.set_title(f'Autoencoder Weights for\n {feature}')
plt.tight_layout()
plt.show()


# list the features with positive weights for each factor:
positive_features = {}
for factor in df_weights.columns:
    positive_features[factor] = df_weights.index[df_weights[factor] > 0].tolist()

# print the features with positive weights for each factor:
for factor, features in positive_features.items():
    print(f"Features with positive weights for {factor}:")
    print(features)
    print()

"""
We see that factors 2 and 3 have the most features with positive weights.
"""
# %% AE: Clusters
# Verify reproducibility by "blindly" identifying c­lusters from the factors (i.e.,
# do the blind c­lusters match with the actual classes?): 
dbscan = DBSCAN(eps=1.80, min_samples=5)
dbscan_labels = dbscan.fit_predict(np.array(encoded_data))
dbscan_labels_unique = set(dbscan_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        ax.scatter(cluster_points[:,0], cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
ax.view_init(elev=35, azim=45)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_zlabel('Factor 3')
ax.set_title('DBSCAN Clustering of the latent variables')
ax.zaxis.labelpad=-3.9
ax.legend()
plt.show()
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
# XY projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"Cluster {label+1}")
axs[0].set_xlabel('Factor 1')
axs[0].set_ylabel('Factor 2')
axs[0].set_title('DBSCAN classes XY Projection')
# XZ projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[1].set_xlabel('Factor 1')
axs[1].set_ylabel('Factor 3')
axs[1].set_title('DBSCAN classes XZ Projection')
# YZ projection:
for label in dbscan_labels_unique:
    if label ==-1:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], marker="x", 
                   label="Noise", color="grey")
    else:
        cluster_points = np.array(encoded_data)[dbscan_labels==label]
        axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[2].set_xlabel('Factor 2')
axs[2].set_ylabel('Factor 3')
axs[2].set_title('DBSCAN classes YZ Projection')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(np.array(encoded_data))
kmeans_labels_unique = set(kmeans_labels)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for label in kmeans_labels_unique:
    cluster_points = encoded_data[kmeans_labels==label]
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
        
fig, axs = plt.subplots(1, 3, figsize=(12, 4))  
# XY projection:
for label in kmeans_labels_unique:
    cluster_points = encoded_data[kmeans_labels==label]
    axs[0].scatter(cluster_points[:,0], cluster_points[:,1], 
                   label=f"Cluster {label+1}")
axs[0].set_xlabel('Factor 1')
axs[0].set_ylabel('Factor 2')
axs[0].set_title('KMEANS classes XY Projection')
# XZ projection:
for label in kmeans_labels_unique:
    cluster_points = encoded_data[kmeans_labels==label]
    axs[1].scatter(cluster_points[:,0], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[1].set_xlabel('Factor 1')
axs[1].set_ylabel('Factor 13')
axs[1].set_title('KMEANS classes XZ Projection')
# YZ projection:
for label in kmeans_labels_unique:
    cluster_points = encoded_data[kmeans_labels==label]
    axs[2].scatter(cluster_points[:,1], cluster_points[:,2], 
                   label=f"Cluster {label+1}")
axs[2].set_xlabel('Factor 2')
axs[2].set_ylabel('Factor 3')
axs[2].set_title('KMEANS classes YZ Projection')
# %% VALIDATING THE AE MODEL

# plot the loss over epochs:
"""
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
"""
#history = autoencoder.fit(X_Factor, X_Factor, epochs=200, batch_size=16, shuffle=True, verbose=0)
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

# split the dataset into training and testing sets:
X_train, X_val = train_test_split(X_Factor, test_size=0.2, random_state=0)

# compile and train the model:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=200, batch_size=16)

# plot the updated loss curve:
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
# %% REGULARIZING THE AE MODEL
# the model is overfitting. Let's try to regularize it:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

# create the encoding layer with L1 regularization
encoded = Dense(encoding_dim, activation='relu', 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
# create the decoding layer
decoded = Dense(num_features, activation='sigmoid')(encoded)

# create the autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)

# compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# train the model:
history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=200, batch_size=16, shuffle=True)

# plot the updated loss curve:
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
# %% CHANGING THE LEARNING RATE OF THE AE MODEL

# change the learning rate:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

# specify the learning rate:
lr = 1 #0.001 default

# create an Adam optimizer with the given learning rate_
optimizer = Adam(lr=lr)

# compile and train the model:
autoencoder.compile(optimizer=optimizer, loss='mse')
history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=200, batch_size=16, shuffle=True)

# plot the updated loss curve:
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
# %% ADDING MORE LAYERS TO THE AE MODEL
# adding more layers:
random.seed(0)        # Python
np.random.seed(0)     # NumPy (which Keras uses)
tf.random.set_seed(0) # TensorFlow

# define encoding layers:
max_nodes = 32
encoded = Dense(max_nodes, activation='relu', 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoded = Dense(np.floor(max_nodes/2), activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(encoded)
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(encoded)

# define decoding layers:
decoded = Dense(np.floor(max_nodes/2), activation='relu')(encoded)
decoded = Dense(max_nodes, activation='relu')(decoded)
decoded = Dense(num_features, activation='sigmoid')(decoded)

# construct the autoencoder model:
autoencoder = Model(input_layer, decoded)

# compile and train the model:
autoencoder.compile(optimizer="Adam", loss='mse')
history = autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=200, batch_size=16, shuffle=True)

# plot the updated loss curve:
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
# %% END
