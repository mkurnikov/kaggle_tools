from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np

from kaggle_tools.utils import nan_tolerant_mean, distance_matrix


class ClusteringImputer():
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


    def fit_transform(self, data):
        if hasattr(data, 'values'):
            data = data.values
        # TODO: add check for type, and warning/error in case of wrong type
        data = data.astype(np.float64)

        self.distance_matrix = distance_matrix(data, self.distance)
        element_to_label = self.clusterer.fit_predict(self.distance_matrix)
        labels = np.unique(element_to_label)

        # doesn't mutate original data
        old_data = np.copy(data)
        for label in labels:
            indices_of_cluster_samples = np.where(element_to_label == label)[0]
            samples = data[indices_of_cluster_samples, :]
            for index_of_sample in indices_of_cluster_samples:
                for col in range(data.shape[1]):
                    if np.isnan(data[index_of_sample, col]):
                        if len(samples[:, col][np.where(~np.isnan(samples[:, col]))]) == 0:
                            data[index_of_sample, col] = nan_tolerant_mean(old_data[:, col])
                        else:
                            data[index_of_sample, col] = nan_tolerant_mean(samples[:, col])
        return data






