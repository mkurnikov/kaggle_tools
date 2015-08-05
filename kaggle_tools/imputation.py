from __future__ import division, print_function

import numpy as np

from kaggle_tools.utils.numeric_utils import nan_tolerant_mean, distance_matrix
from sklearn.base import TransformerMixin, BaseEstimator

class ClusteringImputer(BaseEstimator, TransformerMixin):
    #TODO: imputer doesn't work correctly.
    """
        Clustering-based data imputation. Clustering algorithm have to support distance matrix.

    """

    # TODO: add median, most_frequent statistics as a value for imputation
    def __init__(self, clusterer, distance='euclidean'):
        self.clusterer = clusterer
        if distance != 'euclidean':
            raise ValueError("ClusterImputer support only euclidean distance by now.")
        self.distance = distance
        self.distance_matrix = None


    def set_params(self, **params):
        self.clusterer.set_params(**params)


    def get_params(self, deep=True):
        return self.clusterer.get_params(deep)


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        if hasattr(X, 'values'):
            X = X.values
        # TODO: add check for type, and warning/error in case of wrong type
        X = X.astype(np.float64)

        self.distance_matrix = distance_matrix(X, self.distance)
        element_to_label = self.clusterer.fit_predict(self.distance_matrix)
        labels = np.unique(element_to_label)

        # doesn't mutate original data
        old_data = np.copy(X)
        for label in labels:
            indices_of_cluster_samples = np.where(element_to_label == label)[0]
            samples = X[indices_of_cluster_samples, :]
            for index_of_sample in indices_of_cluster_samples:
                for col in range(X.shape[1]):
                    if np.isnan(X[index_of_sample, col]):
                        if len(samples[:, col][np.where(~np.isnan(samples[:, col]))]) == 0:
                            X[index_of_sample, col] = nan_tolerant_mean(old_data[:, col])
                        else:
                            X[index_of_sample, col] = nan_tolerant_mean(samples[:, col])
        return X







