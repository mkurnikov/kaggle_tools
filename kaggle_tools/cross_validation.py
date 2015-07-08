from __future__ import division, print_function, unicode_literals

from collections import Sized
from collections import namedtuple

import numpy as np
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import indexable
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV

from kaggle_tools.base import is_classifier, clone


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



class MyGridSearchCV(GridSearchCV):
    """
    Differences:
        method one-standard-error.
        training scores.
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=False, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        super(MyGridSearchCV, self).__init__(
            estimator, param_grid, scoring, fit_params, n_jobs, iid,
            False, cv, verbose, pre_dispatch, error_score)
        self.best_ = None


    def _fit(self, X, y, parameter_iterable):
        estimator = self.estimator
        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch

        parallel = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )
        out = parallel(
            delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                    train, test, self.verbose, parameters,
                                    self.fit_params, return_train_score=True, return_parameters=True,
                                    error_score=self.error_score)
                for parameters in parameter_iterable
                for train, test in cv)

        n_fits = len(out)
        n_folds = len(cv)

        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            train_score = 0
            test_score = 0
            all_train_scores = []
            all_test_scores = []
            for this_train_score, this_test_score, this_n_test_samples, \
                _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                # this_n_train_samples = n_samples - this_n_test_samples
                all_train_scores.append(this_train_score)
                all_test_scores.append(this_test_score)
                # if self.iid:
                #     this_test_score *= this_n_test_samples
                #     n_test_samples += this_n_test_samples
                train_score += this_train_score
                test_score += this_test_score
            train_score /= float(n_folds)
            test_score /= float(n_folds)
            # if self.iid:
            #     score /= float(n_test_samples)
            # else:
            #     score /= float(n_folds)
            # scores.append((test_score, parameters))
            grid_scores.append(_CVTrainTestScoreTuple(
                parameters,
                test_score,
                np.array(all_test_scores),
                train_score,
                np.array(all_train_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        self.best_ = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = self.best_.parameters
        self.best_score_ = self.best_.mean_validation_score

    def get_best_one_std(self, std_coeff=1.0):
        if self.best_ is None:
            self.best_ = sorted(self.grid_scores_, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
        best_std = self.best_.cv_validation_scores.std()
        filtered_scores = [score for score in self.grid_scores_
                           if abs(score.mean_validation_score - self.best_.mean_validation_score) < std_coeff * best_std]
        best_one_std = sorted(filtered_scores, key=lambda x: x.mean_training_score)[0]
        return best_one_std


def my_cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False):
    X, y = indexable(X, y)

    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                          train, test, verbose, None,
                                          fit_params, return_train_score=True)
                  for train, test in cv)

    if return_train_score:
        return np.array(scores)[:, [0, 1]]
    else:
        return np.array(scores)[:, [0]]