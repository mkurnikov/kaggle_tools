from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from collections import Sized
import numpy as np
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import indexable
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV

from kaggle_tools.base import is_classifier, clone
from kaggle_tools.grid_search_helpers import _CVTrainTestScoreTuple, CVResult
from kaggle_tools.tools_logging import SklearnToMongo, _get_array_hash

class MyGridSearchCV(GridSearchCV):
    """
    Differences:
        method one-standard-error.
        training scores.
    """

    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=False, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise', logger=None, mongo_collection=None):
        super(MyGridSearchCV, self).__init__(
            estimator, param_grid, scoring, fit_params, n_jobs, iid,
            False, cv, verbose, pre_dispatch, error_score)
        self.best_ = None
        self.logger = logger
        self.mongo_collection = mongo_collection



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
        cv_results = []
        for parameters in parameter_iterable:

            if self.mongo_collection is not None:
                entry = self._check_presence_in_mongo_collection(base_estimator, X, y, cv,
                                                                 self.mongo_collection,
                                                                 parameters, scorer=self.scorer_)
                if entry is not None:
                    scores_ = np.array([entry['scores']['train_scores'],
                                        entry['scores']['test_scores']]).T
                    current_cv_result = CVResult(base_estimator, X, y, cv, parameters,
                                                 scores=scores_,
                                                 scorer=self.scorer_)

                    cv_results.append(current_cv_result)
                    if self.verbose > 0:
                        print(current_cv_result.custom_est_params, current_cv_result.scores[:, [1]].flatten().mean())
                    continue

            out = parallel(
                delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                        train, test, self.verbose, parameters,
                                        self.fit_params, return_train_score=True, return_parameters=True,
                                        error_score=self.error_score)
                    for train, test in cv)
            # print('iteration completed')

            scores = np.array(out)[:, [0, 1]]

            current_cv_result = CVResult(base_estimator, X, y, cv, parameters, scores=scores,
                                         scorer=self.scorer_)
            cv_results.append(current_cv_result)

            if self.logger is not None:
                self.logger.info(current_cv_result)
            if self.mongo_collection is not None:
                self.mongo_collection.insert_one(current_cv_result.to_mongo_repr())

            # print(result.scores.shape)
            if self.verbose > 0:
                print(current_cv_result.custom_est_params, current_cv_result.scores[:, [1]].flatten().mean())

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


    def _check_presence_in_mongo_collection(self, estimator, X, y, cv,
                                            mongo_collection,
                                            params=None, scorer=None):
        data = {
            'X': _get_array_hash(X),
            'y': _get_array_hash(y)
        }

        cv = SklearnToMongo(cv)
        estimator = clone(estimator)
        estimator.set_params(**params)
        estimator = SklearnToMongo(estimator)
        cv_config_json_obj = {
            'estimator': estimator,
            'cv': cv,
            'data': data,
            'scorer': SklearnToMongo(scorer)
        }
        entry = mongo_collection.find_one(cv_config_json_obj)
        return entry


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
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import KFold

    X, y = make_regression(random_state=2)
    cv = KFold(len(y), n_folds=5, random_state=1)

    from pymongo import MongoClient
    client = MongoClient()

    collection = client['test']['grid_search']

    from sklearn.linear_model import Ridge
    clf = Ridge()
    grid_search = MyGridSearchCV(clf, {'alpha': [0.1, 0.01]}, cv=cv,
                                 mongo_collection=collection, verbose=1)
    grid_search.fit(X, y)

    print(grid_search.grid_scores_)