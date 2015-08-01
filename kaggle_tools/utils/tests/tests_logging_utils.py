from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from xgboost import XGBRegressor

from kaggle_tools.utils.logging_utils import MongoCollectionWrapper, MongoSerializer


def main():
    pipeline = Pipeline([
        # ('aa', TransformFeatureSet(settings.CATEGORICAL, StringToInt())),
        ('linear', LinearRegression())
    ])
    # repr_ = MongoDBRepresentation(pipeline).to_json()
    from sklearn.cross_validation import LeaveOneLabelOut

    serializer = MongoSerializer(ignored_fields=['n_estimators'])
    cv = KFold(10, n_folds=4)
    print(serializer.serialize(cv))
    repr_ = serializer.serialize(pipeline)
    #
    def func(param1, param2, *args):
        pass
    #
    print(serializer.serialize(func))
    print(serializer.serialize(XGBRegressor()))
    class MyClass():
        @classmethod
        def scorer(cls, X_test, y_test):
            pass

    print(serializer.serialize(MyClass.scorer))
    print(serializer.serialize(MyClass().scorer))


if __name__ == '__main__':
    main()