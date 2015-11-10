from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

import numpy as np
import pandas as pd

from sklearn.utils.testing import assert_equal, assert_array_equal
from kaggle_tools.cross_validation import RepeatedKFold, KFoldBy


def test_repeated_kfold():
    X = [1, 3, 2, 1, 1, 2]
    cv = RepeatedKFold(n_folds=3, random_state=1)

    res = []
    res.append((np.array([0, 3, 4, 5]), np.array([1, 2])))
    res.append((np.array([1, 2, 3, 5]), np.array([0, 4])))
    res.append((np.array([0, 1, 2, 4]), np.array([3, 5])))
    res.append((np.array([0, 1, 3, 5]), np.array([2, 4])))
    res.append((np.array([1, 2, 3, 4]), np.array([0, 5])))
    res.append((np.array([0, 2, 4, 5]), np.array([1, 3])))

    for i, (train, test) in enumerate(cv.split(X)):
        # print(train, test)
        assert_array_equal(res[i][0], train)
        assert_array_equal(res[i][1], test)


def test_kfold_by_for_pandas_dataframe():
    df = pd.DataFrame(data={'col1': [1, 2, 3],
                       'col2': [5, 6, 7]},
                      index=[4, 3, 2])
    cv = KFoldBy(by='col1', n_folds=3)

    res = [
        (pd.DataFrame(data={'col1': [1], 'col2': [5]}, index=[4]),
         pd.DataFrame(data={'col1': [2, 3], 'col2': [6, 7]}, index=[3, 2])),
        (pd.DataFrame(data={'col1': [2], 'col2': [6]}, index=[3]),
         pd.DataFrame(data={'col1': [1, 3], 'col2': [5, 7]}, index=[4, 2])),
        (pd.DataFrame(data={'col1': [3], 'col2': [7]}, index=[2]),
         pd.DataFrame(data={'col1': [1, 2], 'col2': [5, 6]}, index=[4, 3]))
        # pd.DataFrame(data={'col1': [2], 'col2': [6]}, index=[3]),
        # pd.DataFrame(data={'col1': [3], 'col2': [7]}, index=[2]),
        # pd.DataFrame(data={'col1': [4], 'col2': [8]}, index=[1]),
    ]
    for i, (train, test) in enumerate(cv.split(df)):
        assert_equal(res[i][0], train)
        assert_equal(res[i][1], test)


if __name__ == '__main__':
    test_repeated_kfold()
    test_kfold_by_for_pandas_dataframe()











