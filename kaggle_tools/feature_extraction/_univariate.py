from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
	# noinspection PyUnresolvedReferences
	from py3compatibility import *


import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.utils.random import sample_without_replacement, check_random_state
import warnings


class DescriptiveStatistics(BaseEstimator, TransformerMixin):
    """Count descriptive statistics of target for every unique value of feature.

    statistic           callable that supports interface func(arr, axis=0)

    all_but_n_ratio     if not None, then serves as a multiplier for size of set of y values corresponding to feature value.
                        Can't be used with all_but_n_rows. Default in None (corresponds to 1.0)

    all_but_n_rows      if not None, then corresponds to the number of rows to be removed from set of y values
                        corresponding to feature value.
                        Can't be used with all_but_n_ratio.

    random_state        Random seed that is used to subsampling.
    """
    def __init__(self, statistic=np.mean, all_but_n_ratio=None, all_but_n_rows=None, random_state=None,
                 noise=False):
        if not hasattr(statistic, '__call__'):
            raise ValueError("'statistic' parameter should be callable.")
        self.statistic = statistic

        if all_but_n_ratio is not None and all_but_n_rows is not None:
            raise ValueError("'all_but_n_ratio' and 'all_but_n_rows' can not both have not-None value.")

        self.all_but_n_ratio = all_but_n_ratio
        self.all_but_n_rows = all_but_n_rows

        self.rng = check_random_state(random_state)
        self.noise = noise
        self.mappings = None


    def fit(self, X, y=None):
        X, y = check_X_y(X, y, copy=True)
        self.n_features = X.shape[1]

        # mask = np.zeros((X.shape[0],), dtype=np.bool_)
        self.mappings = {}
        for f in range(self.n_features):
            unique_vals, indices = np.unique(X[:, f], return_inverse=True)
            val_stat_mapping = {}
            for ind, unique_val in enumerate(unique_vals):
                # mask[:] = False
                # mask[X[:, f] == unique_val] = True
                y_for_value = y[indices == ind]
                y_for_value_size = y_for_value.shape[0]
                if self.all_but_n_ratio is not None:
                    if y_for_value_size * self.all_but_n_ratio > 1:
                        y_for_value_indices = sample_without_replacement(y_for_value_size,
                                                                         y_for_value_size * self.all_but_n_ratio,
                                                                         random_state=self.rng)
                        y_for_value = y_for_value[y_for_value_indices]
                    else:
                        warnings.warn('Not enough rows to count statistic for feature = {}, value = {}'
                                      .format(f, unique_val))

                if self.all_but_n_rows is not None:
                    if y_for_value_size - self.all_but_n_rows > 1:
                        y_for_value_indices = sample_without_replacement(y_for_value_size,
                                                                         y_for_value_size - self.all_but_n_rows,
                                                                         random_state=self.rng)
                        y_for_value = y_for_value[y_for_value_indices]
                    else:
                        warnings.warn('Not enough rows to count statistic for feature = {}, value = {}'
                                      .format(f, unique_val))

                val_stat_mapping[unique_val] = self.statistic(y_for_value)
            self.mappings[f] = val_stat_mapping
        return self


    def transform(self, X):
        X = check_array(X, copy=True)
        if self.n_features != X.shape[1]:
            raise ValueError('Inconsistent number of features, given: {}, should be: {}'
                             .format(X.shape[1], self.n_features))
        X = X.astype(np.float64)

        for f in range(self.n_features):
            unique_values, indices = np.unique(X[:, f], return_inverse=True)
            if set(unique_values) - set(self.mappings[f].keys()) != set():
                raise ValueError('Transformed array has values that are not presented in original array. ')
            for ind, val in enumerate(unique_values):
                values = np.ones_like((X[indices == ind, f])) * self.mappings[f][val]
                if self.noise:
                    values *= np.random.normal(loc=1.0, scale=0.01, size=values.shape)
                X[indices == ind, f] = values

        return X



class NonlinearTransformationFeatures(BaseEstimator, TransformerMixin):
    """Apply univariate (typically non-linear) transformation to all presented features.

    transformation          Transformation to be applied.
                            Can be from ['sqrt', 'log'] or callable.

    """
    def __init__(self, transformation='sqrt'):
        if not hasattr(transformation, '__call__') and transformation not in ['sqrt', 'log']:
            raise ValueError('{t} transformation is not supported. Use some of {list}'.format(t=transformation,
                                                                                                  list=['sqrt', 'log']))
        self.transformation = transformation

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, copy=True)

        for f in range(X.shape[1]):
            if hasattr(self.transformation, '__call__'):
                X[:, f] = self.transformation(X[:, f])

            elif self.transformation == 'sqrt':
                X[:, f] = np.sqrt(X[:, f])

            elif self.transformation == 'log':
                col = X[:, f]
                mask = np.isclose(col, 0)
                col = np.log(col)
                col[mask] = 0
                X[:, f] = col

            else:
                ValueError('Unsupported method {}'.format(self.transformation))

        return X


class Identity(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

