from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
    # noinspection PyUnresolvedReferences
    from py3compatibility import *

import numpy as np
from sklearn.base import clone
from sklearn.externals.joblib import delayed
from sklearn.grid_search import _fit_and_score


from ._base import MyBaseGridSearch
from ._helpers import CVResult
from kaggle_tools.utils import logging_utils


def custom_fit_and_score_(estimator, X, y, scorer,
                          train, test, verbose, parameters,
                          return_train_score=False, error_score='raise'):
    return _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters, None,
                          return_train_score=return_train_score, return_parameters=True, error_score=error_score)


class MyGridSearchCV(MyBaseGridSearch):
    def __init__(self, estimator, param_grid,
                 scoring=None, n_jobs=1, cv=None, iid=True,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 logger=None, collection_wrapper=None):
        super(MyGridSearchCV, self).__init__(estimator, param_grid,
                                             scoring=scoring, n_jobs=n_jobs, cv=cv, iid=iid,
                                             verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score)

        self.logger = logger
        self.collection_wrapper = collection_wrapper


    def _fit_and_cv_result(self, estimator, X, y, parameters, cv,
                           parallel=None, scorer=None, verbose=0):
        if self.collection_wrapper is not None:
            existing_cv_result = self._extract_existing_cv_result(estimator, X, y, cv,
                                                                  parameters, scorer=self.scorer_)
            if existing_cv_result is not None:
                return existing_cv_result


        out = parallel(
            delayed(custom_fit_and_score_)(clone(estimator), X, y, scorer,
                                    train, test, verbose, parameters,
                                    return_train_score=True, error_score=self.error_score)
                for train, test in cv)

        scores = np.array(out)[:, [0, 1]]

        current_cv_result = CVResult(estimator, X, y, cv, parameters,
                                     scores=scores, scorer=self.scorer_)

        if self.logger is not None:
            message = str(current_cv_result)
            message += '\n'
            message += logging_utils.STARLINE
            self.logger.info(message)

        if self.collection_wrapper is not None:
            self.collection_wrapper.insert_cv_result(current_cv_result)

        return current_cv_result


    def _extract_existing_cv_result(self, estimator, X, y, cv,
                                            params=None, scorer=None):
        estimator = clone(estimator)
        # from kaggle_tools.tools_logging import MongoCollectionWrapper
        # MongoCollectionWrapper.check_presence_in_mongo_collection_early_stop()
        entry = (self.collection_wrapper
                 .check_presence_in_mongo_collection(estimator, X, y, cv,
                                                params, scorer=scorer))
        if entry is not None:
            scores_ = np.array([entry['scores']['train_scores'],
                                entry['scores']['test_scores']]).T
            current_cv_result = CVResult(estimator, X, y, cv, params,
                                         scores=scores_,
                                         scorer=self.scorer_)
            return current_cv_result
        return None


if __name__ == '__main__':
    from kaggle_tools.grid_search.tests.tests_grid_search import main
    main()












