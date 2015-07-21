from __future__ import division, print_function

import pandas as pd
import numpy as np
import warnings
from sklearn.utils.validation import check_array
from sklearn.utils import deprecated

from kaggle_tools.base import BaseEstimator, TransformerMixin


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

        return X_integers#.reshape((n_samples, 1))
