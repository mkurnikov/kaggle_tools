from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor

from sklearn.utils.testing import assert_equal

from kaggle_tools.utils import pipeline_utils


def test_find_final_estimator_path_prefix_as_xgboost():
    pipeline = Pipeline([
        ('11', Pipeline([
            ('22', XGBRegressor())
        ]))
    ])
    prefix = pipeline_utils.find_final_estimator_param_prefix(pipeline)[0]
    assert_equal(prefix, '11__22__')


def test_find_final_estimator_path_prefix_as_linear_regression():
    pipeline = Pipeline([
        ('11', Pipeline([
            ('22', LinearRegression())
        ]))
    ])
    prefix = pipeline_utils.find_final_estimator_param_prefix(pipeline)[0]
    assert_equal(prefix, '11__22__')


def test_find_final_estimator_path_prefix_as_linear_classification():
    pipeline = Pipeline([
        ('11', Pipeline([
            ('22', LogisticRegression())
        ]))
    ])
    prefix = pipeline_utils.find_final_estimator_param_prefix(pipeline)[0]
    assert_equal(prefix, '11__22__')


def test_get_final_estimator():
    pipeline = Pipeline([
        ('11', Pipeline([
            ('22', LogisticRegression())
        ]))
    ])

    final_ = pipeline_utils.get_final_estimator(pipeline)
    assert_equal(final_.__class__, LogisticRegression)




















