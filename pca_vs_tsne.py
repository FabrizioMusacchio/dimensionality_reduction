"""
A script to compare PCA and t-SNE on different datasets.

author: Fabrizio Musacchio
date: April 5, 2023
"""

# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# %% EXAMPLE 1
# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.title("PCA")
plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.title("t-SNE")
plt.tight_layout()
plt.savefig('pca_vs_tsne_1.png', dpi=200)
plt.show()
# %% EXAMPLE 2
# Load the handwritten digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10')
plt.colorbar(label='Digit')
plt.title("PCA")

plt.subplot(122)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10')
plt.colorbar(label='Digit')
plt.title("t-SNE")

plt.tight_layout()
plt.savefig('pca_vs_tsne_2.png', dpi=200)
plt.show()
# %% EXAMPLE 3
# Generate the Swiss Roll dataset
X, color = make_swiss_roll(n_samples=1500, noise=0.5, random_state=42)

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

# Plot the results
fig = plt.figure(figsize=(12, 6))

# Plot PCA results
ax1 = fig.add_subplot(121)
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='jet')
ax1.set_title("PCA")

# Plot t-SNE results
ax2 = fig.add_subplot(122)
ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='jet')
ax2.set_title("t-SNE")

plt.tight_layout()
plt.savefig('pca_vs_tsne_3.png', dpi=200)
plt.show()

# plot the swiss roll:
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='jet')
ax.view_init(8, -75)
plt.tight_layout()
plt.savefig('pca_vs_tsne_3_swissroll.png', dpi=200)
plt.show()

# %% END
