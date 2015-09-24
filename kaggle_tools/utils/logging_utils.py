from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import FeatureUnion, Pipeline
# import json

from collections import OrderedDict
def pipeline_to_dict(pipeline):
    if not isinstance(pipeline, BaseEstimator):
        raise ValueError('pipeline has to be an instance of BaseEstimator.')

    if isinstance(pipeline, (ClassifierMixin, RegressorMixin)):
        return str(pipeline)

    res = OrderedDict()
    for name, estimator in pipeline.steps:
        if isinstance(estimator, FeatureUnion):
            res[name] = feature_union_to_array(estimator)
            continue

        else:
            res[name] = pipeline_to_dict(estimator)
            continue

    return res



def feature_union_to_array(union):
    if not isinstance(union, FeatureUnion):
        raise ValueError('union has to be an instance of FeatureUnion.')

    features = []
    for name, transformer in union.transformer_list:
        if isinstance(transformer, FeatureUnion):
            features.append((name, feature_union_to_array(transformer)))

        else:
            features.append(name)

    return features


STARLINE = '*' * 120


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor
    from sklearn.preprocessing import PolynomialFeatures

    from kaggle_tools.feature_extraction import Identity
    pipeline = Pipeline([
        ('Features', FeatureUnion([
            ('Identity', Identity()),
            ('Polynomials', PolynomialFeatures())
        ])),
        ('Estimator', Pipeline([
            ('XGBRegressor', XGBRegressor())
        ]))
    ])

    import json
    s = json.dumps(pipeline_to_dict(pipeline), indent=2)
    print(s)
    # s = s.replace('\n', '\\n')
    # print(s)
















