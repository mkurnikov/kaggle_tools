from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np


def nan_tolerant_euclidean(v1, v2):
    """
        Euclidean distance for vectors with missing values.
    """
    return sum([(x1 - x2) ** 2 for x1, x2 in zip(v1, v2)
                if not np.isnan(x1) and not np.isnan(x2)])


def nan_tolerant_mean(data, **kwargs):
    """
        Mean of array with missing values.
    """
    mean = np.ma.masked_array(data, mask=np.isnan(data)).mean(**kwargs)
    # if all elements of original array was np.nan, mean will contain full masked array
    mean = np.ma.filled(mean, fill_value=np.nan)
    return mean


def distance_matrix(data, distance='euclidean'):
    """
        Compute distance matrix for the provided dataset.
    """
    if distance == 'euclidean':
        dist_matrix = np.ndarray((data.shape[0], data.shape[0]), np.float64)
        for i in range(dist_matrix.shape[0]):
            for j in range(dist_matrix.shape[1]):
                dist_matrix[i, j] = nan_tolerant_euclidean(data[i, :], data[j, :])
        return dist_matrix
    raise NotImplementedError


def pprint_cross_val_scores(scores):
    if type(scores) == list:
        scores = np.array(scores)
    print("%0.8f (+/-%0.05f)" % (scores.mean(), scores.std()))
