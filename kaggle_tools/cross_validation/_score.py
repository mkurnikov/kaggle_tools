from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

import numpy as np
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import indexable
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _fit_and_predict
from sklearn.base import is_classifier, clone

from kaggle_tools.grid_search import CVResult
from kaggle_tools.utils import logging_utils


def my_cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score=False,
                       logger=None, score_all_at_once=False):
    X, y = indexable(X, y)
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)


    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    if not score_all_at_once:
        scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                              train, test, verbose, None,
                                              fit_params, return_train_score=True)
                      for train, test in cv)

        cv_result = CVResult(estimator, X, y, cv, scores=(np.array(scores)[:, [0]],
                                                          np.array(scores)[:, [1]]))
        if logger is not None:
            message = str(cv_result)
            message += '\n'
            message += logging_utils.STARLINE
            logger.info(message)

        if return_train_score:
            return np.array(scores)[:, [0, 1]]
        else:
            return np.array(scores)[:, [1]]

    else:
        predictions = np.empty_like(y, dtype=y.dtype)
        scores = parallel(delayed(_fit_and_predict)(clone(estimator), X, y,
                                              train, test, verbose, fit_params)
                      for train, test in cv)
        for preds, test_idx in scores:
            predictions[test_idx] = preds


        class SuggorateEstimator(estimator.__class__):
            def __init__(self):
                pass

            def predict(self, X):
                return predictions

        score = scorer(SuggorateEstimator(), X, y)

        cv_result = CVResult(estimator, X, y, cv, scores=score)
        if logger is not None:
            logger.info(cv_result)

        return score