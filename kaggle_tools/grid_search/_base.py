from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from abc import ABCMeta, abstractmethod
from sklearn.externals import six
from sklearn.grid_search import Sized
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier
from sklearn.utils.validation import indexable, _num_samples
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel
from sklearn.grid_search import ParameterGrid, _check_param_grid

from kaggle_tools.grid_search._helpers import _CVTrainTestScoreTuple
from kaggle_tools.utils.misc_utils import pprint_cross_val_scores


class MyBaseGridSearch(six.with_metaclass(ABCMeta, BaseEstimator,
                                          MetaEstimatorMixin)):
    """Rewritten version of GridSearchCV from sklearn library. Lack of refit() logic, some cleanings,
    more flexibility for subclasses.
    Adds logging functionality, training scores, one-standard-error method.
    """
    def __init__(self, estimator, param_grid,
                 scoring=None, n_jobs=1, cv=None, iid=True,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise'):

        self.estimator = estimator

        _check_param_grid(param_grid)
        self.parameters_iterable = ParameterGrid(param_grid)
        self.scorer_ = check_scoring(self.estimator, scoring=scoring)

        self.n_jobs = n_jobs
        self.iid = iid
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score


    def fit(self, X, y=None):
        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))

        if self.verbose > 0:
            if isinstance(self.parameters_iterable, Sized):
                n_candidates = len(self.parameters_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=self.pre_dispatch
        )
        cv_results = []
        for parameters in self.parameters_iterable:
            #main logic of class stored inside _fit_and_cv_result() method
            cv_result = self._fit_and_cv_result(clone(self.estimator), X, y, parameters, cv,
                                                parallel=parallel, scorer=self.scorer_, verbose=self.verbose)

            if self.verbose > 0:
                self._print_params_and_result(cv_result)

            cv_results.append(cv_result)


        grid_scores = list()
        for cvresult in cv_results:
            train_scores = cvresult.scores[:, [0]].flatten()
            test_scores = cvresult.scores[:, [1]].flatten()
            grid_scores.append(_CVTrainTestScoreTuple(
                cvresult.custom_est_params,
                test_scores.mean(), test_scores,
                train_scores.mean(), train_scores))

        # Store the computed scores
        self.grid_scores_ = grid_scores

        self.best_ = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = self.best_.parameters
        self.best_score_ = self.best_.mean_validation_score



    @abstractmethod
    def _fit_and_cv_result(self, estimator, X, y, parameters, cv,
                           parallel=None, scorer=None, verbose=0):
        return None


    def _print_params_and_result(self, cv_result):
        print(cv_result.custom_est_params, end=' ')
        pprint_cross_val_scores(cv_result.scores[:, [1]].flatten())


    def get_best_one_std(self, std_coeff=1.0):
        if self.best_ is None:
            self.best_ = sorted(self.grid_scores_, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
        best_std = self.best_.cv_validation_scores.std()
        filtered_scores = [score for score in self.grid_scores_
                           if abs(score.mean_validation_score - self.best_.mean_validation_score) < std_coeff * best_std]
        best_one_std = sorted(filtered_scores, key=lambda x: x.mean_training_score)[0]
        return best_one_std









