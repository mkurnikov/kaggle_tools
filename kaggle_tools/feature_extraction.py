from __future__ import division, print_function

from itertools import combinations
import sys
import numpy as np
from scipy import sparse

from sklearn.utils import deprecated
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureColumnsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        if columns is None:
            raise AttributeError("Columns for features haven't been provided.")

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


@deprecated('Use standard sklearn.feature_extraction.PolynomialFeatures instead.')
class HighOrderFeatures(BaseEstimator, TransformerMixin):
    """
        Look at (similar idea)
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html


    """

    def __init__(self, degree=2, feature_locations=None):
        self.degree = degree
        self.feature_locations = feature_locations

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # new_data = []

        if hasattr(X, 'index'):
            X = X.values

        n_samples = X.shape[0]
        n_features = X.shape[1]
        if not self.feature_locations:
            self.feature_locations = range(n_features)

        combs = list(combinations(self.feature_locations, self.degree))

        import time
        new_data = np.empty((X.shape[0], len(combs)))
        # before = time.time()
        samples = range(0, n_samples)
        for col, indices in enumerate(combs):
            # feature_combinations_values = []
            arr = X[:, indices]
            sorted_idx = np.lexsort(arr.T)
            sorted_arr = arr[sorted_idx, :]

            df1 = np.diff(sorted_arr, axis=0)
            df2 = np.append([False], np.any(df1 != 0, axis=1), axis=0)
            labels = df2.cumsum(axis=0)
            values = np.zeros_like(labels)
            values[sorted_idx] = labels
            new_data[:, col] = values

            # for row, sample_val in enumerate(X[:, indices]):
            # for row in samples:

                # new_data[row, col] = hashlib.md5(sample_val.tostring()).digest()
                # new_data[row, col] = hash(sample_val.tostring()) % ((sys.maxsize + 1) * 2)
                # new_data[row, col] = hash(tuple(X[row, indices])) % ((sys.maxsize + 1) * 2)
                # new_data[row, col] = hash(X[row, indices].tostring()) % ((sys.maxsize + 1) * 2)
            #     new_data[row, col] = hash(tuple(sample_val)) % ((sys.maxsize + 1) * 2)

            # print(time.time() - before)
            # print(col)
            # if col % 20 == 0:
            #     print(time.time() - before)
            #     before = time.time()
            # 0.577078104019
                # feature_combinations_values.append(
                #     # str(tuple(v))
                #     hash(tuple(sample_val)) % ((sys.maxsize + 1) * 2)
                # )
            # new_data.append(feature_combinations_values)
        # return np.array(new_data).T
        return new_data


@deprecated('Use standard sklearn.preprocessing.data.OneHotEncoder instead.')
class SparseOneHotEncoder(BaseEstimator, TransformerMixin):
    """
        OneHotEncoder based on scipy.sparse

        look on
            http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html
            http://scikit-learn.org/stable/modules/feature_extraction.html#dict-feature-extraction
        and comparison at the bottom.

        Difference:
        OneHotEncoder takes as input categorical values encoded as integers - you can get them from LabelEncoder.

        DictVectorizer expects data as a list of dictionaries, where each dictionary is a data row with column names as keys:
        [ { 'foo': 1, 'bar': 'z' },
      { 'foo': 2, 'bar': 'a' },
      { 'foo': 3, 'bar': 'c' } ]

      After vectorizing and saving as CSV it would look like this:
      foo,bar=z,bar=a,bar=c
        1,1,0,0
        2,0,1,0
        3,0,0,1
        Notice the column names and that DictVectorizer doesn't touch numeric values.

        http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/
    """


    def __init__(self, keymap=None):
        self.keymap = keymap


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
