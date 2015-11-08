from __future__ import division, print_function

import pandas as pd
import numpy as np
import warnings
from sklearn.utils.validation import check_array
from sklearn.utils import deprecated

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureColumnsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        if columns is None:
            raise AttributeError("Columns for features haven't been provided.")

        if type(columns) != list:
            raise AttributeError('columns argument has to be array.')

        if len(columns) == 0:
            raise AttributeError("Length of columns array is zero.")
        self.columns = columns

        self.idtype = None
        if isinstance(columns[0], str) or isinstance(columns[0], unicode):
            self.idtype = 'pandas'
        elif isinstance(columns[0], int):
            self.idtype = 'numpy'

        if self.idtype is None:
            raise AttributeError("Unknown type of column's id: {}".format(type(columns[0])))


    def fit(self, X, y=None):
        return self


    def transform(self, X):
        if self.idtype == 'pandas':
            return X[self.columns]

        # In case of numpy
        raise NotImplementedError


class StringToInt(BaseEstimator, TransformerMixin):
    """
        Map Series object's values from arbitrary objects to unique integer. It's useful for sklearn algorithms.

        nan_strategy    Specify, how to treat NaN values in the array. Available strategies:

                        'mask' - create masked array out of the origin, and perform tranformation on it,
                            NaNs will remain NaNs.
                        'value' - treat NaN value as separate value.

    """

    def __init__(self, nan_strategy='mask'):
        if nan_strategy not in ['mask', 'value']:
            raise ValueError("Incorrect NaN strategy: {strategy}. "
                             "Only supported NaN strategies is 'mask' and 'value'."
                             .format(strategy=nan_strategy))
        self.nan_strategy = nan_strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = check_array(X, copy=True, dtype=None, force_all_finite=False, ensure_2d=True)

        if X.shape[1] != 1:
            raise ValueError('Input array has to be 1d, got {} instead.'.format(X_integers.shape))

        n_samples = X.shape[0]
        # it can process np.nan as a separate value out of box.
        unique_vals, X_integers = np.unique(X, return_inverse=True)
        nan_value = -1
        for i, unique_val in enumerate(unique_vals):
            if type(unique_val) == float and np.isnan(unique_val):
                nan_value = i
                break

        X_integers = X_integers.astype(np.float)
        if self.nan_strategy == 'mask':
            X_integers[X_integers == nan_value] = np.nan


        return X_integers.reshape((n_samples, 1))
