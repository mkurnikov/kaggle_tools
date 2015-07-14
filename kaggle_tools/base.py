from __future__ import division, print_function

from sklearn.base import *


class TransformerMixin(TransformerMixin):
    def fit(self, X, y=None):
        """
            Helper method for IDE to enforce method signature of transformers.
        """
        raise NotImplementedError


    def transform(self, X):
        """
            Helper method for IDE to enforce method signature of transformers.
        """
        raise NotImplementedError