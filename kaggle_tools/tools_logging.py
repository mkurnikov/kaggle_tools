from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import _PartitionIterator, KFold
from collections import Iterable, OrderedDict

import numpy as np
def SklearnToMongo(obj):
    return MongoDBRepresentation(obj).to_json()

class MongoDBRepresentation(object):
    def __init__(self, obj):
        self.obj = obj
        self.representation = None

        if isinstance(obj, Pipeline):
            self.representation = self._pipeline(obj)

        elif isinstance(obj, FeatureUnion):
            self.representation = self._feature_union(obj)

        elif isinstance(obj, BaseEstimator):
            self.representation = self._estimator(obj)

        elif isinstance(obj, _PartitionIterator):
            self.representation = self._cv(obj)

        elif isinstance(obj, np.ndarray):
            if len(obj.shape) > 1:
                raise NotImplementedError
            self.representation = list(obj)

        elif isinstance(obj, Iterable):
            self.representation = list(obj)

        elif hasattr(obj, '__call__'):
            self.representation = obj.func_name

        else:
            raise NotImplementedError


    def _feature_union(self, obj):
        steps = []
        for step_id, estimator in obj.steps:
            steps.append({
                'step_id': step_id,
                'transformer': MongoDBRepresentation(estimator).to_json()
            })
        json_obj = {
            'name': obj.__class__.__name__,
            'steps': steps
        }
        return json_obj


    def _pipeline(self, obj):
        steps = []
        for step_id, estimator in obj.steps:
            steps.append({
                          'step_id': step_id,
                          'estimator': MongoDBRepresentation(estimator).to_json()
            })
        json_obj = {
            'name': obj.__class__.__name__,
            'steps': steps
        }
        return json_obj


    def _estimator(self, estimator):
        json_obj = {
            'name': estimator.__class__.__name__,
            'params': estimator.get_params(),
        }
        return json_obj


    def _cv(self, cv):
        params = {
            'random_state': cv.random_state,
            'n': cv.n,
        }
        if isinstance(cv, KFold):
            params['n_folds'] = cv.n_folds
            params['shuffle'] = cv.shuffle

        # TODO: add RepeatedKFold
        # if isinstance(cv, RepeatedKFold):
        #     pass

        json_obj = {
            'name': cv.__class__.__name__,
            'params': params,
        }
        return json_obj


    def __repr__(self):
        return str(self.representation)

    def __str__(self):
        return str(self.representation)

    def to_json(self):
        return self.representation



def _get_array_hash(arr):
    if hasattr(arr, 'index'):
        arr_hashable = arr.values.copy()
    else:
        arr_hashable = arr

    arr_hashable.flags.writeable = False
    hash_ = hash(arr_hashable.data)
    arr_hashable.flags.writeable = True
    return hash_


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    pipeline = Pipeline([
        ('linear', LinearRegression())
    ])
    # repr_ = MongoDBRepresentation(pipeline).to_json()
    cv = KFold(10, n_folds=4)
    repr_ = SklearnToMongo(cv)
    # repr_ = {
    #     'Maxim': {
    #         '1': True,
    #         '2' : False
    #     },
    #     'name': 'Linear'
    # }
    import json
    # repr_ =  {'params': {'copy_X': True, 'normalize': False, 'n_jobs': 1, 'fit_intercept': True}, 'name': 'LinearRegression'}

    # parsed = json.loads(str(repr_))
    print(json.dumps(repr_, indent=4, sort_keys=True))


