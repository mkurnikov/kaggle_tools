from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from collections import Sized, namedtuple
import numpy as np
from numbers import Number
from sklearn.base import clone, is_classifier
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import indexable
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.grid_search import GridSearchCV

# from kaggle_tools.base import is_classifier, clone
from kaggle_tools.utils.misc_utils import _get_array_hash
from kaggle_tools.utils.misc_utils import pprint_cross_val_scores, \
    _get_pprinted_cross_val_scores, _get_pprinted_mean


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


class CVResult(object):
    def __init__(self, estimator, X, y, cv, custom_est_params=None, scores=None, scorer=None):
        self.estimator = clone(estimator)
        if custom_est_params is not None:
            self.estimator.set_params(**custom_est_params)
        # if self.estimator.get_params().
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
        msg = ''
        msg += str(self.estimator) + '\n'
        msg += str((self.X.shape, self.y.shape)) + '\n'
        msg += str(self.custom_est_params) + '\n'
        msg += str(self.cv) + '\n'

        if self.score_type == 'array':
            train_score = _get_pprinted_cross_val_scores(self.scores[:, [0]].flatten())
            test_score = _get_pprinted_cross_val_scores(self.scores[:, [1]].flatten())
            msg += str((train_score, test_score))

        elif self.score_type == 'number':
            score = _get_pprinted_mean(self.scores)
            msg += str(score)

        else:
            raise ValueError

        return msg
