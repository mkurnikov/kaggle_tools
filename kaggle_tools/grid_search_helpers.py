from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

import numpy as np
from collections import namedtuple
from collections import Iterable
from numbers import Number
from kaggle_tools.utils import _get_pprinted_cross_val_scores, _get_pprinted_mean, _get_pprinted_std
from kaggle_tools.tools_logging import SklearnToMongo, _get_array_hash

class _CVTrainTestScoreTuple(namedtuple('_CVTrainTestScoreTuple',
                                        ('parameters',
                                        'mean_validation_score',
                                        'cv_validation_scores',
                                        'mean_training_score',
                                        'cv_training_scores'))):
    __slots__ = ()

    def __repr__(self):
        return "mean: {0:.5f}, std: {1:.5f}, params: {2}".format(
            self.mean_validation_score,
            np.std(self.cv_validation_scores),
            self.parameters)

from sklearn.base import clone
class CVResult(object):
    def __init__(self, estimator, X, y, cv, custom_est_params=None, scores=None, scorer=None):
        self.estimator = clone(estimator)
        self.estimator.set_params(**custom_est_params)
        self.X = X
        self.y = y
        self.custom_est_params = custom_est_params
        self.cv = cv
        self.scores = scores
        self.scorer = scorer
        if isinstance(self.scores, (tuple, np.ndarray)):
            self.score_type = 'array'
        elif isinstance(self.scores, Number):
            self.score_type = 'number'
        else:
            raise ValueError

    def __repr__(self):
        if self.score_type == 'array':
            train_score = _get_pprinted_cross_val_scores(self.scores[:, [0]].flatten())
            test_score = _get_pprinted_cross_val_scores(self.scores[:, [1]].flatten())

        elif self.score_type == 'number':
            score = _get_pprinted_mean(self.scores)

        else:
            raise ValueError

        msg = ''
        msg += str(self.estimator) + '\n'
        msg += str((self.X.shape, self.y.shape)) + '\n'
        msg += str(self.custom_est_params) + '\n'
        msg += str(self.cv) + '\n'
        if self.score_type == 'array':
            msg += str((train_score, test_score))
        else:
            msg += str(score)
        return msg


    def to_mongo_repr(self):
        scores = None
        if self.score_type == 'array':
            scores = {
                'score_type': 'array',
                'train_scores': SklearnToMongo(self.scores[:, [0]].flatten()),
                'test_scores': SklearnToMongo(self.scores[:, [1]].flatten())
            }
        elif self.score_type == 'number':
            scores = {
                'score_type': 'number',
                'score': SklearnToMongo(self.scores)
            }

        data = {
            'X': _get_array_hash(self.X),
            'y': _get_array_hash(self.y)
        }
        json_obj = {
            'estimator': SklearnToMongo(self.estimator),
            'data': data,
            'cv': SklearnToMongo(self.cv),
            'scores': scores,
            'custom_params': self.custom_est_params,
            'scorer': SklearnToMongo(self.scorer)
        }
        return json_obj