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

from kaggle_tools.grid_search import CVResult, _CVTrainTestScoreTuple


class MyGridSearchCV(GridSearchCV):
    """
    Differences:
        method one-standard-error.
        training scores.
        logging to mongo/files.
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=False, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise', logger=None, mongo_collection_wrapper=None):
        super(MyGridSearchCV, self).__init__(
            estimator, param_grid, scoring, fit_params, n_jobs, iid,
            False, cv, verbose, pre_dispatch, error_score)
        self.best_ = None
        self.logger = logger
        self.mongo_collection_wrapper = mongo_collection_wrapper


    def _fit(self, X, y, parameter_iterable):
        # estimator = self.estimator
        cv = self.cv

        # for custom internal scoring_
        self.scorer_ = self._make_scorer(self.estimator, self.scoring)
        # self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        #create out-of-fold eval_set for xgboost gridsearch. Here does nothing.
        X, y = self._custom_split(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(self.estimator))

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
        cv_results = []
        for parameters in parameter_iterable:
            # print(_get_array_hash(X))
            # print(X)
            if self.mongo_collection_wrapper is not None:
                existing_cv_result = self._extract_existing_cv_result(base_estimator, X, y, cv,
                                                                      parameters, scorer=self.scorer_)
                # entry = MongoUtils.check_presence_in_mongo_collection(base_estimator, X, y, cv,
                #                                                  self.mongo_collection,
                #                                                  parameters, scorer=self.scorer_)
                # if entry is not None:
                #     scores_ = np.array([entry['scores']['train_scores'],
                #                         entry['scores']['test_scores']]).T
                #     current_cv_result = CVResult(base_estimator, X, y, cv, parameters,
                #                                  scores=scores_,
                #                                  scorer=self.scorer_)
                if existing_cv_result is not None:
                    cv_results.append(existing_cv_result)

                    if self.verbose > 0:
                        self._print_params_and_result(existing_cv_result)

                    continue

            return_parameters = True
            out = parallel(
                delayed(self._get_custom_fit_and_score())(clone(base_estimator), X, y, self.scorer_,
                                        train, test, self.verbose, parameters,
                                        self.fit_params, return_train_score=True,
                                        return_parameters=return_parameters,
                                        error_score=self.error_score)
                    for train, test in cv)
            # print('iteration completed')
            if return_parameters:
                processed_params = self._process_parameters(np.array(out)[:, 4])
                if processed_params is not None:
                    parameters = processed_params

            scores = np.array(out)[:, [0, 1]]

            current_cv_result = CVResult(base_estimator, X, y, cv, parameters, scores=scores,
                                         scorer=self.scorer_)
            cv_results.append(current_cv_result)

            if self.logger is not None:
                self.logger.info(current_cv_result)
            if self.mongo_collection_wrapper is not None:
                self.mongo_collection_wrapper.insert_cv_result(current_cv_result)

            # print(result.scores.shape)
            if self.verbose > 0:
                self._print_params_and_result(current_cv_result)

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


    def _get_custom_fit_and_score(self):
        return _fit_and_score
        # return _fit_and_score(estimator, X, y, scorer, train, test, verbose,
        #                       parameters, fit_params, return_train_score, return_parameters,
        #                       error_score)

    # @staticmethod

    def _process_parameters(self, params_arr):
        return None


    def _make_scorer(self, estimator, scoring):
        return check_scoring(self.estimator, scoring=self.scoring)


    def _custom_split(self, X, y):
        return X, y


    def _print_params_and_result(self, cv_result):
        print(cv_result.custom_est_params, end=' ')
        pprint_cross_val_scores(cv_result.scores[:, [1]].flatten())


    def _extract_existing_cv_result(self, estimator, X, y, cv,
                                            params=None, scorer=None):
        estimator = clone(estimator)
        # from kaggle_tools.tools_logging import MongoCollectionWrapper
        # MongoCollectionWrapper.check_presence_in_mongo_collection_early_stop()
        entry = (self.mongo_collection_wrapper
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


    def get_best_one_std(self, std_coeff=1.0):
        if self.best_ is None:
            self.best_ = sorted(self.grid_scores_, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
        best_std = self.best_.cv_validation_scores.std()
        filtered_scores = [score for score in self.grid_scores_
                           if abs(score.mean_validation_score - self.best_.mean_validation_score) < std_coeff * best_std]
        best_one_std = sorted(filtered_scores, key=lambda x: x.mean_training_score)[0]
        return best_one_std


if __name__ == '__main__':
    from kaggle_tools.grid_search.tests.tests_grid_search import main
    main()