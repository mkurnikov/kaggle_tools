from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import six


class DataFrameMapper(BaseEstimator, TransformerMixin):
    """
    Map Pandas data frame column subsets to their own
    sklearn transformation.

    """


    def __init__(self, features, return_df=True):
        """
        Params:

        features    a list of pairs. The first element is the pandas column
                    selector. This can be a string (for one column) or a list
                    of strings. The second element is an object that supports
                    sklearn's transform interface.
        return_df   perform inverse mapping or return plain numpy array
        """
        self.features = features
        self.return_df = return_df


    def _get_col_subset(self, X, cols):
        """
        Get a subset of columns from the given table X.

        X       a Pandas dataframe; the table to select columns from
        cols    a string or list of strings representing the columns
                to select

        Returns a numpy array with the data from the selected columns
        """
        if isinstance(cols, six.string_types):
            cols = [cols]

        return X[cols]


    def fit(self, X, y=None):
        """
        Fit a transformation from the pipeline

        X       the data to fit
        """
        if not hasattr(X, 'index'):
            raise ValueError('X is not a DataFrame or Series. DataFrameMapper can operate only on Pandas objects.')

        for columns, transformer in self.features:
            if transformer is not None:
                transformer.fit(self._get_col_subset(X, columns))
        return self


    def transform(self, X):
        """
        Transform the given data. Assumes that fit has already been called.

        X       the data to transform
        """

        if len(self.features) == 0:
            return X

        sub_dfs = []
        for columns, transformer in self.features:
            # columns could be a string or list of
            # strings; we don't care because pandas
            # will handle either.
            if transformer is not None:
                subset = self._get_col_subset(X, columns)
                transformed_subset = transformer.transform(subset)
            else:
                transformed_subset = self._get_col_subset(X, columns)

            if hasattr(transformed_subset, 'toarray'):
                # sparse arrays should be converted to regular arrays
                # for hstack.
                transformed_subset = transformed_subset.toarray()

            # TODO - possible inefficiency: creating python list with dataset size
            if len(transformed_subset.shape) == 1:
                transformed_subset = np.array([transformed_subset]).T

            # TODO - add possibility for columns = list
            if self.return_df:
                if isinstance(columns, list):
                    raise NotImplementedError('return_df=True will not work right with features as list.')

                original_feature_name = columns
                new_features = []
                for i in range(transformed_subset.shape[1]):
                    new_features.append('{fname}_{transformation}_{idx}'.format(fname=original_feature_name,
                                                                                transformation=transformer.__class__.__name__,
                                                                                idx=i))

                transformed_subset = pd.DataFrame(transformed_subset,
                                                  index=X.index,
                                                  columns=new_features)

            sub_dfs.append(transformed_subset)

        # combine the feature outputs into one array.
        # at this point we lose track of which features
        # were created from which input columns, so it's
        # assumed that that doesn't matter to the model.

        if self.return_df:
            return pd.concat(sub_dfs, axis=1, join='inner')
        else:
            return np.hstack(sub_dfs)
