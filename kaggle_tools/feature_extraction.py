from __future__ import division, absolute_import, print_function, unicode_literals
from itertools import combinations
import sys
import numpy as np
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin


class HighOrderFeaturesGenerator(BaseEstimator, TransformerMixin):
    # TODO: check correctness
    # TODO: optimize hashings
    # TODO: check for compatibility with feature union
    def __init__(self, degree=2, feature_locations=None):
        self.degree = degree
        self.feature_locations = feature_locations


    def fit_transform(self, X, y=None, **fit_params):
        new_data = []

        n_features = X.shape[1]
        if not self.feature_locations:
            self.feature_locations = range(n_features)

        combs = combinations(self.feature_locations, self.degree)
        for indices in combs:
            feature_combinations_values = [hash(tuple(v)) % ((sys.maxsize + 1) * 2) for v in X[:, indices]]
            new_data.append(feature_combinations_values)
        return np.array(new_data).T


class SparseOneHotEncoder(BaseEstimator, TransformerMixin):
    """
        OneHotEncoder based on scipy.sparse
    """
    def __init__(self, keymap=None):
        self.keymap = keymap


    # TODO: rewrite to numpy
    # TODO: check correctness
    # TODO: perform refactoring (learn scipy.sparse before)
    def fit_transform(self, X, y=None, **fit_params):
        if self.keymap is None:
            self.keymap = []
            for col in X.T:
                uniques = set(list(col))
                self.keymap.append(dict((key, i) for i, key in enumerate(uniques)))
        total_pts = X.shape[0]
        outdat = []

        for feature_number, feature_values in enumerate(X.T):
            km = self.keymap[feature_number]
            num_labels = len(km)
            spmat = sparse.lil_matrix((total_pts, num_labels))

            for j, value in enumerate(feature_values):
                if value in km:
                    spmat[j, km[value]] = 1
            outdat.append(spmat)
        outdat = sparse.hstack(outdat).tocsr()

        raise NotImplementedError("encoder returns scipy.sparse array, not numpy")
        return outdat, self.keymap