from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

from sklearn.utils.testing import assert_equal

from xgboost import XGBRegressor

from kaggle_tools.utils.mongo_utils import MongoCollectionWrapper, MongoSerializer
from kaggle_tools.utils import pipeline_utils
from kaggle_tools.utils.mongo_utils import _path_from_prefix


def test_path_from_prefix():
    pipeline = Pipeline([
        ('11', Pipeline([
            ('22', XGBRegressor())
        ]))
    ])

    prefix = pipeline_utils.find_final_estimator_param_prefix(pipeline)[0]
    assert_equal(_path_from_prefix(pipeline, prefix), 'steps.0.11.steps.0.22.')


def test_mongo_serializer_cv():
    serializer = MongoSerializer()

    cv = KFold(10, n_folds=4)
    assert_equal(serializer.serialize(cv),
                 'sklearn.cross_validation.KFold(n=10, n_folds=4, shuffle=False, random_state=None)')


def test_mongo_serializer_func():
    serializer = MongoSerializer()

    def func(param1, param2, *args):
        pass
    assert_equal(serializer.serialize(func), 'func(param1, param2, *args)')


def test_mongo_serializer_classmethod():
    serializer = MongoSerializer()

    class MyClass():
        @classmethod
        def scorer(cls, X_test, y_test):
            pass
    assert_equal(serializer.serialize(MyClass.scorer), 'MyClass.scorer(X_test, y_test)')