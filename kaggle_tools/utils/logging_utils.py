from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
import inspect
from collections import Callable

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import _PartitionIterator

from kaggle_tools.grid_search import CVResult
from kaggle_tools.utils import misc_utils


class MongoSerializer(object):
    """MongoDB equivalent to __repr__ method for logging.
    Use in conjunction with MongoCollection object (alternative to Logger object).
    """
    def __init__(self, ignored_fields=None):
        if ignored_fields is None:
            ignored_fields = []
        self.ignored_fields = set(ignored_fields)


    def serialize(self, obj):
        if isinstance(obj, Pipeline):
            return self._pipeline(obj)

        elif isinstance(obj, CVResult):
            return self._cv_result_object(obj)

        elif isinstance(obj, FeatureUnion):
            return self._feature_union(obj)

        elif isinstance(obj, BaseEstimator):
            return self._estimator(obj)

        elif isinstance(obj, _PartitionIterator):
            return self._cv(obj)

        elif isinstance(obj, np.ndarray):
            return self._ndarray(obj)

        elif isinstance(obj, dict):
            return self._dict(obj)

        elif isinstance(obj, (list, tuple)):
            return self._list(obj)

        elif isinstance(obj, Callable):
            return self._callable(obj)

        else:
            return obj


    def _pipeline(self, pipeline):
        steps = []
        for step_id, estimator in pipeline.steps:
            steps.append({
                step_id: self.serialize(estimator)
            })
        json_obj = {
            'name': pipeline.__class__.__name__,
            'steps': steps
        }
        return json_obj


    def _feature_union(self, union):
        steps = []
        for step_id, transformer in union.transformer_list:
            steps.append({
                step_id: self.serialize(transformer)
            })
        json_obj = {
            'name': union.__class__.__name__,
            'steps': steps
        }
        return json_obj


    def _estimator(self, estimator):
        json_obj = {
            'name': estimator.__class__.__name__,
            'params': self.serialize(estimator.get_params(deep=False)),
        }
        return json_obj


    def _dict(self, dict_obj):
        json_obj = {}
        for k, v in dict_obj.iteritems():
            if k not in self.ignored_fields:
                json_obj[k] = self.serialize(v)

        return json_obj


    def _ndarray(self, ndarray_obj):
        return self.serialize(list(ndarray_obj))


    def _callable(self, obj):
        if inspect.isfunction(obj):
            #user-defined (unbound) functions
            args, varargs, kwargs, _ = inspect.getargspec(obj)
            arguments = []
            arguments.extend(args)
            if varargs is not None:
                arguments.append('*' + varargs)
            if kwargs is not None:
                arguments.append('**' + kwargs)
            return '{name}({args})'.format(name=obj.__name__,
                                           args=', '.join(arguments))

        elif inspect.ismethod(obj):
            #class methods, instance methods
            args, varargs, kwargs, _ = inspect.getargspec(obj)
            bounded_obj_name = obj.__self__.__name__
            args = args[1:]
            arguments = []
            arguments.extend(args)
            if varargs is not None:
                arguments.append('*' + varargs)
            if kwargs is not None:
                arguments.append('**' + kwargs)

            return '{bounded}.{name}({args})'.format(bounded=bounded_obj_name,
                                                     name=obj.__name__,
                                                     args=', '.join(arguments))
        else:
            return repr(obj)


    def _list(self, list_obj):
        if not isinstance(list_obj, list):
            list_obj = list(list_obj)
        json_obj = []
        for el in list_obj:
            json_obj.append(self.serialize(el))
        return json_obj


    def _cv(self, cv):
        # TODO: create much better representation using something like non-existent cv.get_params()
        return repr(cv)


    def _cv_result_object(self, obj):
        scores = None
        if obj.score_type == 'array':
            scores = {
                'score_type': 'array',
                'train_scores': self.serialize(obj.scores[:, [0]].flatten()),
                'test_scores': self.serialize(obj.scores[:, [1]].flatten())
            }
        elif obj.score_type == 'number':
            scores = {
                'score_type': 'number',
                'score': self.serialize(obj.scores)
            }

        data = {
            'X': misc_utils._get_array_hash(obj.X),
            'y': misc_utils._get_array_hash(obj.y)
        }
        json_obj = {
            'estimator': self.serialize(obj.estimator),
            'data': data,
            'cv': self.serialize(obj.cv),
            'scores': scores,
            'custom_params': self.serialize(obj.custom_est_params),
            'scorer': self.serialize(obj.scorer)
        }
        return json_obj



class MongoCollectionWrapper(object):
    def __init__(self, serializer=None, collection=None):
        # serializer validation
        if serializer is None:
            raise ValueError('MongoSerializer object must be specified.')

        if not isinstance(serializer, MongoSerializer):
            raise ValueError('serializer has wrong class {}'.format(serializer.__class__.__name__))

        # collection validation
        if collection is None:
            raise ValueError('db collection must be specified.')

        self.serializer = serializer
        self.collection = collection


    def insert_cv_result(self, cv_result):
        self.collection.insert_one(self.serializer.serialize(cv_result))


    def check_presence_in_mongo_collection(self, estimator, X, y, cv,
                                            params=None, scorer=None):
        data = {
            'X': misc_utils._get_array_hash(X),
            'y': misc_utils._get_array_hash(y)
        }

        cv = self.serializer.serialize(cv)
        estimator = clone(estimator)
        estimator.set_params(**params)
        estimator = self.serializer.serialize(estimator)
        cv_config_json_obj = {
            'estimator': estimator,
            'cv': cv,
            'data': data,
            'scorer': self.serializer.serialize(scorer)
        }
        entry = self.collection.find_one(cv_config_json_obj)
        return entry


    def check_presence_in_mongo_collection_early_stop(self, estimator, X, y, cv,
                                            params=None, scorer=None):
        data = {
            'X': misc_utils._get_array_hash(X),
            'y': misc_utils._get_array_hash(y)
        }

        cv = self.serializer.serialize(cv)
        estimator = clone(estimator)
        estimator.set_params(**params)
        estimator = self.serializer.serialize(estimator)
        print(estimator)
        cv_config_json_obj = {
            'estimator': estimator,
            'cv': cv,
            'data': data,
            'scorer': self.serializer.serialize(scorer)
        }
        entry = self.collection.find_one(cv_config_json_obj)
        return entry


if __name__ == '__main__':
    from kaggle_tools.utils.tests import tests_logging_utils
    tests_logging_utils.main()