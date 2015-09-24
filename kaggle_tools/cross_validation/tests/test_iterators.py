from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np

from sklearn.utils.testing import assert_allclose
from kaggle_tools.cross_validation import RepeatedKFold


def test_repeated_kfold():
    X = [1, 3, 2, 1, 1, 2]
    cv = RepeatedKFold(len(X), random_state=1)

    res = []
    res.append((np.array([0, 3, 4, 5]), np.array([1, 2])))
    res.append((np.array([1, 2, 3, 5]), np.array([0, 4])))
    res.append((np.array([0, 1, 2, 4]), np.array([3, 5])))
    res.append((np.array([0, 1, 2, 5]), np.array([3, 4])))
    res.append((np.array([0, 1, 3, 4]), np.array([2, 5])))
    res.append((np.array([2, 3, 4, 5]), np.array([0, 1])))

    for i, (train, test) in enumerate(cv):
        assert_allclose(res[i][0], train)
        assert_allclose(res[i][1], test)