from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np

from sklearn.utils.testing import assert_allclose

from kaggle_tools.feature_extraction import DescriptiveStatistics


def test_descriptive_mean():
    X = np.array([[1, 1, 2, 2, 2],
              [5, 4, 3, 2, 1]]).T
    y = np.array([1, 2, 3, 4, 5])
    X = DescriptiveStatistics().fit_transform(X, y)

    res = np.array([[1.5,  1.], [1.5, 2.], [4., 3.], [4., 4.], [4., 5.]])
    assert_allclose(X, res)


