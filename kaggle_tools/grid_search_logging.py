from __future__ import division, print_function

import six
if six.PY2:
	# noinspection PyUnresolvedReferences
	from py3compatibility import *


from collections import Sized
from collections import namedtuple

import numpy as np
from sklearn.cross_validation import _fit_and_score
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import indexable
from sklearn.cross_validation import check_cv, check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.grid_search import GridSearchCV

from sklearn.base import is_classifier, clone



import json
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
        results = []
        for parameters in parameter_iterable:
            out = parallel(
                delayed(_fit_and_score)(clone(base_estimator), X, y, self.scorer_,
                                        train, test, self.verbose, parameters,
                                        self.fit_params, return_train_score=True, return_parameters=True,
                                        error_score=self.error_score)
                    for train, test in cv)

            scores = np.array(out)[:, [0, 1]]
            result = CVResult(base_estimator, X, y, cv, parameters, scores=scores)
            results.append(result)
            if self.logger is not None:
                self.logger.info(result)
            if self.mongo_collection is not None:
                self.mongo_collection.insert_one(result.to_mongo_repr())

            # print(result.scores.shape)
            print(result.custom_est_params, result.scores[:, [1]].flatten().mean())

        grid_scores = list()
        for cvresult in results:
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


    def get_best_one_std(self, std_coeff=1.0):
        if self.best_ is None:
            self.best_ = sorted(self.grid_scores_, key=lambda x: x.mean_validation_score,
                  reverse=True)[0]
        best_std = self.best_.cv_validation_scores.std()
        filtered_scores = [score for score in self.grid_scores_
                           if abs(score.mean_validation_score - self.best_.mean_validation_score) < std_coeff * best_std]
        best_one_std = sorted(filtered_scores, key=lambda x: x.mean_training_score)[0]
        return best_one_std




from sklearn.cross_validation import _fit_and_predict
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
        logger.info(cv_result)

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

        # if logger is not None:
        #     logger.info(estimator)
        #     logger.info(cv)
        return score

        # logger.info(scores)
    # # print(estimator.get_params())
    # print(cv)
    # print(scores)





if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import KFold

    X, y = make_regression()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = LinearRegression()
    # clf.fit(X_train, y_train)

    import logging
    logging.basicConfig(filename='regression.log', level=logging.INFO)
    logger = logging.getLogger('log.regression')
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)

    cv = KFold(len(y), n_folds=5, random_state=1)
    # print(my_cross_val_score(clf, X, y, cv=cv,
    #                          logger=logger, score_all_at_once=False))

    from pymongo import MongoClient
    client = MongoClient()
    collection = client['hazard']['grid_search']

    from sklearn.linear_model import Lasso, Ridge
    clf = Ridge()
    grid_search = MyGridSearchCV(clf, {'alpha': [0.1, 0.01]}, cv=cv,
                                 mongo_collection=collection)
    grid_search.fit(X, y)


    print(grid_search.grid_scores_)













