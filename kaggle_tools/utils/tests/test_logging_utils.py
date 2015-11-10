from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six

if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

from sklearn.utils.testing import assert_equal
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import FeatureUnion, Pipeline

from kaggle_tools.feature_extraction import Identity

from kaggle_tools.utils import logging_utils


def test_pipeline_to_dict():
    pipeline = Pipeline([
        ('Features', FeatureUnion([
            ('Identity', Identity()),
            ('Polynomials', PolynomialFeatures())
        ])),
        ('Estimator', Pipeline([
            ('LinearRegression', LinearRegression())
        ]))
    ])

    d = logging_utils.pipeline_to_dict(pipeline)
    true_res = "OrderedDict([('Features', ['Identity', 'Polynomials']), ('Estimator', OrderedDict([('LinearRegression', " \
               "'LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)')]))])"
    assert_equal(str(d), true_res)
