from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six

if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state


class RepeatedKFold(_BaseKFold):
    def __init__(self, n_folds=3, n_repeats=2, random_state=None):
        super(RepeatedKFold, self).__init__(n_folds, False, random_state)
        self.n_repeats = n_repeats
        self.random_state = check_random_state(random_state)

    def _iter_test_indices(self, X=None, y=None, labels=None):
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        # check_random_state(self.random_state).shuffle(indices)
        for _ in range(self.n_repeats):
            shuffled_indices = indices.copy()
            self.random_state.shuffle(shuffled_indices)
            # shuffle_state = check_random_state(self.random_state.randint(0))
            # shuffled_indices = shuffle_state.shuffle(indices)

            n_folds = self.n_folds
            fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=np.int)
            fold_sizes[:n_samples % n_folds] += 1
            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                yield shuffled_indices[start:stop]
                current = stop
        #
        # for _ in range(self.n_repeats):
        #     # self.rng.shuffle()
        #     for idxs in super(RepeatedKFold, self)._iter_test_indices(X):
        #         yield idxs


from sklearn.model_selection._split import _BaseKFold, KFold
from sklearn.utils.validation import _num_samples
import numpy as np


class KFoldBy(_BaseKFold):
    """
        Create KFold cross-validation splits based on specific column in pandas dataset.
        (for numpy array use labels)

        by      if string -> column name in pandas dataframe
    """

    def __init__(self, by, presort=False, n_folds=3, shuffle=False, random_state=None):
        super(KFoldBy, self).__init__(n_folds, shuffle, random_state)
        self.by = by
        self.presort = presort


    def _iter_test_masks(self, X=None, y=None, labels=None):
        if not hasattr(X, 'iloc'):
            raise ValueError('X has to be pandas array. Given {}'.format(type(X)))

        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_folds = self.n_folds
        fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n_samples % n_folds] += 1

        col_values = X[self.by].copy()
        if self.presort:
            col_values.sort_values(inplace=True)


            # col_values.
