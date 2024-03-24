import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)
class PCA:
  def __init__(self, n_components) -> None:
    self.n_components = n_components
    self.eigen_values = None
    self.eigen_vectors = None
    self.X_std = None

  # 1 - Data normalization
  def Standardize_data(self,X):
    mean = np.sum(X, axis = 0) / X.shape[0]
    std = (np.sum((X - mean)**2, axis=0) / (X.shape[0]-1)) ** 0.5
    X_std = (X - mean)/std
    return X_std

  # 2 - Covariance matrix computation
  def covariance(self, X):
    cov = (X.T @ X)/(X.shape[0] - 1)
    return cov

  # 3 - Computation of eigenvalues and eigenvectors
  def eig_val_vec(self, X_std):
    eigen_values, eigen_vectors =  np.linalg.eig(self.covariance(X_std))
    return eigen_values, eigen_vectors.T

  # 4 - Fit the data
  def fit(self, X):
    self.X_std = self.Standardize_data(X)
    self.eigen_values, self.eigen_vectors = self.eig_val_vec(self.X_std)


  # 5 - Projection of the dataset
  def transform(self):
    P = self.eigen_vectors[ : self.n_components, :]
    X_proj = self.Standardize_data(X).dot(P.T) # Projection matrix

    # Analysis's plot
    plt.title(f"PC1 vs PC2")
    plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y) # c=y i.e take the color function of the target y
    plt.xlabel('PC1'); plt.xticks([])
    plt.ylabel('PC2'); plt.yticks([])
    plt.show()
    return X_proj
iris = load_iris()
X = iris['data']
y = iris['target']
my_pca = PCA(n_components=2)
my_pca.fit(X)
new_X = my_pca.transform()
#new_X.shape
