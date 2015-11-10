from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six

if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

import json
import numpy as np
import inspect
from copy import deepcopy
from collections import Callable
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import BaseCrossValidator

from kaggle_tools.grid_search import CVResult
from kaggle_tools.utils import misc_utils

from kaggle_tools.utils import pipeline_utils


def _path_from_prefix(estimator, prefix):
    prefixes = prefix.split('__')
    path = ''
    for prefix in prefixes:
        if isinstance(estimator, Pipeline):
            for i, (name, step) in enumerate(estimator.steps):
                if name == prefix:
                    path += 'steps' + '.'
                    path += str(i) + '.'
                    path += name + '.'
                    estimator = step
    return path

#
# from inspect import signature
# import warnings
#
# from sklearn.base import _pprint
#
#
# class CVSerializer(object):
#     def _get_param_names(cls):
#         """Get parameter names for the estimator"""
#         # fetch the constructor or the original constructor before
#         # deprecation wrapping if any
#         init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
#         if init is object.__init__:
#             # No explicit constructor to introspect
#             return []
#
#         # introspect the constructor arguments to find the model parameters
#         # to represent
#         init_signature = signature(init)
#         # Consider the constructor parameters excluding 'self'
#         parameters = [p for p in init_signature.parameters.values()
#                       if p.name != 'self' and p.kind != p.VAR_KEYWORD]
#         for p in parameters:
#             if p.kind == p.VAR_POSITIONAL:
#                 raise RuntimeError("scikit-learn estimators should always "
#                                    "specify their parameters in the signature"
#                                    " of their __init__ (no varargs)."
#                                    " %s with constructor %s doesn't "
#                                    " follow this convention."
#                                    % (cls, init_signature))
#         # Extract and sort argument names excluding 'self'
#         return sorted([p.name for p in parameters])
#
#
#     def get_params(self, deep=True):
#         """Get parameters for this estimator.
#
#         Parameters
#         ----------
#         deep: boolean, optional
#             If True, will return the parameters for this estimator and
#             contained subobjects that are estimators.
#
#         Returns
#         -------
#         params : mapping of string to any
#             Parameter names mapped to their values.
#         """
#         out = dict()
#         for key in self._get_param_names():
#             # We need deprecation warnings to always be on in order to
#             # catch deprecated param values.
#             # This is set in utils/__init__.py but it gets overwritten
#             # when running under python3 somehow.
#             warnings.simplefilter("always", DeprecationWarning)
#             try:
#                 with warnings.catch_warnings(record=True) as w:
#                     value = getattr(self, key, None)
#                 if len(w) and w[0].category == DeprecationWarning:
#                     # if the parameter is deprecated, don't show it
#                     continue
#             finally:
#                 warnings.filters.pop(0)
#
#             # XXX: should we rather test if instance of estimator?
#             if deep and hasattr(value, 'get_params'):
#                 deep_items = value.get_params().items()
#                 out.update((key + '__' + k, val) for k, val in deep_items)
#             out[key] = value
#         return out
#
#     def serialize(self, cv):
#         class_name = self.__class__.__name__
#         return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
#                                                offset=len(class_name),),)


class MongoSerializer(object):
    """MongoDB equivalent to __repr__ method for logging.
    Use in conjunction with MongoCollection object (alternative to Logger object).
    """

    def __init__(self, ignored_fields=None):
        if ignored_fields is None:
            ignored_fields = []
        self.ignored_fields = set(ignored_fields)


    def add_ignored_field(self, field):
        self.ignored_fields.add(field)


    def serialize(self, obj):
        if isinstance(obj, Pipeline):
            return self._pipeline(obj)

        # elif isinstance(obj, BaseSubmittion):
        #     return self._submission(obj)
        elif hasattr(obj, 'json_'):
            return obj.json_

        elif isinstance(obj, CVResult):
            return self._cv_result_object(obj)

        elif isinstance(obj, FeatureUnion):
            return self._feature_union(obj)

        elif isinstance(obj, BaseEstimator):
            return self._estimator(obj)

        elif isinstance(obj, BaseCrossValidator):
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
            warnings.warn('Haven\'t found serializer for {} object, use repr() instead.'
                          .format(obj), RuntimeWarning)
            return repr(obj)


    def _submission(self, submission_obj):
        json_obj = {
            'project_submission_id': submission_obj.project_submission_id_,
            'specific_submission_id': submission_obj.submission_id,
            'pipeline': self.serialize(submission_obj.pipeline),
            'cv_scores': self.serialize(submission_obj.cv_scores),
            'submission_score': submission_obj.submission_score
        }
        return json_obj


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
        for k, v in dict_obj.items():
            if k not in self.ignored_fields:
                json_obj[k] = self.serialize(v)

        return json_obj


    def _ndarray(self, ndarray_obj):
        return self.serialize(list(ndarray_obj))


    def _callable(self, obj):
        if inspect.isfunction(obj):
            # user-defined (unbound) functions
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
            # class methods, instance methods
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
        json_cv_result = self.serializer.serialize(cv_result)
        # print(json.dumps(json_cv_result, indent=2))
        # raise SystemExit(1)
        self.collection.insert_one(json_cv_result)


    def insert_submission(self, submission):
        self.collection.insert_one(self.serializer.serialize(submission))


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
            'scorer': self.serializer.serialize(scorer),
        }
        entry = self.collection.find_one(cv_config_json_obj)
        return entry


    def check_presence_in_mongo_collection_early_stop(self, estimator, X, y, cv,
                                                      params=None, scorer=None):
        data = {
            'X': misc_utils._get_array_hash(X),
            'y': misc_utils._get_array_hash(y)
        }
        serializer = deepcopy(self.serializer)
        # serializer.add_ignored_field('n_estimators')

        cv = serializer.serialize(cv)

        estimator = clone(estimator)
        estimator.set_params(**params)
        # estimator_json = serializer.serialize(estimator)
        # print(estimator)
        est_params = serializer.serialize(pipeline_utils.get_final_estimator(estimator).get_params())
        n_ests_key = None
        for p in est_params:
            if 'n_estimators' in p:
                n_ests_key = p
        if n_ests_key is not None:
            del est_params[n_ests_key]

        est_name = serializer.serialize(pipeline_utils.get_final_estimator(estimator).__class__.__name__)

        prefix = pipeline_utils.find_final_estimator_param_prefix(estimator)[0]
        # print(prefix)
        est_path = _path_from_prefix(estimator, prefix)
        # print(est_path)

        est_params_json = {}
        for est_param in est_params:
            est_params_json['estimator.' + est_path + 'params.' + est_param] = est_params[est_param]
            # 'estimator.' + est_path + 'params': est_params,
        cv_config_json_obj = {
            # 'estimator': estimator,
            'estimator.' + est_path + 'name': est_name,
            'cv': cv,
            'data': data,
            'scorer': serializer.serialize(scorer),
        }
        cv_config_json_obj.update(est_params_json)
        # print(json.dumps(cv_config_json_obj, indent=2))
        entry = self.collection.find_one(cv_config_json_obj)
        return entry


if __name__ == '__main__':
    from kaggle_tools.utils.tests import tests_mongo_utils

    tests_mongo_utils.main()
