from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import time
import numbers
import warnings
import xgboost as xgb
import numpy as np

from sklearn.base import clone
from sklearn.externals.joblib import delayed
from sklearn.cross_validation import _safe_split, FitFailedWarning, _score, train_test_split, _num_samples
from sklearn.externals.joblib import logger
from sklearn.utils.validation import check_random_state

from kaggle_tools.utils import pipeline_utils
from kaggle_tools.grid_search._base import MyBaseGridSearch
from kaggle_tools.grid_search._grid_search import custom_fit_and_score_
from kaggle_tools.grid_search._helpers import CVResult


def xgb_early_stop_fit_and_score(estimator, X, y, scorer,
                                 train, test, verbose, parameters,
                                 return_train_score=False, error_score='raise',
                                 hold_out_size=0.1, early_stopping_rounds=10, verbose_eval=False, eval_metric=None,
                                 random_state=None, maximize_score=False):
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    X_train, X_hold_out, y_train, y_hold_out = train_test_split(X_train, y_train,
                                                                test_size=hold_out_size,
                                                                random_state=random_state)
    # X_hold_out, y_hold_out = X_test, y_test
    #
    # X_train, X_hold_out = X_train.iloc[5000:], X_train.iloc[:5000]
    # y_train, y_hold_out = y_train.iloc[5000:], y_train.iloc[:5000]

    eval_set = [(X_train, y_train), (X_hold_out, y_hold_out)]
    xgb_param_prefix = pipeline_utils.find_final_estimator_param_prefix(estimator)[0]
    fit_params = {
        xgb_param_prefix + 'eval_set': eval_set,
        xgb_param_prefix + 'early_stopping_rounds': early_stopping_rounds,
        xgb_param_prefix + 'eval_metric': eval_metric,
        xgb_param_prefix + 'verbose': verbose_eval,
        xgb_param_prefix + 'maximize_score': maximize_score
    }

    try:
        estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)"
                             )
    else:
        test_score = _score(estimator, X_test, y_test, scorer)
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)

    scoring_time = time.time() - start_time

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(X_test), scoring_time])

    parameters = parameters.copy()
    parameters.update({
        xgb_param_prefix + 'n_estimators': pipeline_utils.get_final_estimator(estimator).best_iteration
    })
    ret.append(parameters)
    return ret


class XGBEarlyStopGridSearchCV(MyBaseGridSearch):
    def __init__(self, estimator, param_grid,
                 scoring=None, n_jobs=1, cv=None, iid=True,
                 verbose=0, pre_dispatch='2*n_jobs', error_score='raise',
                 logger=None, collection_wrapper=None,
                 early_stopping_rounds=10, hold_out_size=0.1, eval_metric=None, random_state=None,
                 verbose_eval=False, maximize_score=None):
        super(XGBEarlyStopGridSearchCV, self).__init__(estimator, param_grid,
                                     scoring=scoring, n_jobs=n_jobs, cv=cv, iid=iid,
                                     verbose=verbose, pre_dispatch=pre_dispatch, error_score=error_score)

        final_ = pipeline_utils.get_final_estimator(estimator)
        if not isinstance(final_, xgb.XGBModel):
            raise ValueError('{} can be used only for xgboost models. Current estimator class is {}'
                             .format(self.__class__.__name__, final_.__class__.__name__))

        self.logger = logger
        self.collection_wrapper = collection_wrapper
        self.early_stopping_rounds = early_stopping_rounds
        self.hold_out_size = hold_out_size
        self.eval_metric = eval_metric
        self.rng = check_random_state(random_state)
        self.verbose_eval = verbose_eval
        self.maximize_score = maximize_score

        # def scorer(estimator, X_test, y_test):
        #     best_iteration = pipeline_utils.get_final_estimator(estimator).best_iteration
        #     print(pipeline_utils.get_final_estimator(estimator).__class__)
        #     preds = pipeline_utils.get_final_estimator(estimator).predict(X_test)
        #     print('custom scorer call')
        #     print(preds.shape)
        #     return scoring(preds, y_test)
        # self.scorer_ = scorer


    def _fit_and_cv_result(self, estimator, X, y, parameters, cv,
                           parallel=None, scorer=None, verbose=0):
        if self.collection_wrapper is not None:
            existing_cv_result = self._extract_existing_cv_result(estimator, X, y, cv,
                                                                  parameters, scorer=self.scorer_)
            if existing_cv_result is not None:
                return existing_cv_result

        out = parallel(
            delayed(xgb_early_stop_fit_and_score)(clone(estimator), X, y, scorer,
                                           train, test, verbose, parameters,
                                           return_train_score=True, error_score=self.error_score,
                                           hold_out_size=self.hold_out_size, early_stopping_rounds=self.early_stopping_rounds,
                                           verbose_eval=self.verbose_eval, eval_metric=self.eval_metric,
                                                  random_state=self.rng, maximize_score=self.maximize_score)
            for train, test in cv)

        scores = np.array(out)[:, [0, 1]]
        params_arr = np.array(out)[:, [4]].flatten()
        #
        processed_params = params_arr[0].copy() #example of keys
        n_ests_key = None
        for key in processed_params:
            if 'n_estimators' in key:
                n_ests_key = key
        n_estimators = np.mean([params[n_ests_key] for params in params_arr.flatten()])
        processed_params.update({n_ests_key: int(n_estimators)})


        current_cv_result = CVResult(estimator, X, y, cv, processed_params,
                                     scores=scores, scorer=self.scorer_)
        if self.logger is not None:
            self.logger.info(current_cv_result)
        if self.collection_wrapper is not None:
            self.collection_wrapper.insert_cv_result(current_cv_result)

        return current_cv_result


    def _extract_existing_cv_result(self, estimator, X, y, cv,
                                            params=None, scorer=None):
        estimator = clone(estimator)
        # from kaggle_tools.tools_logging import MongoCollectionWrapper
        # MongoCollectionWrapper.check_presence_in_mongo_collection_early_stop()
        entry = (self.collection_wrapper
                 .check_presence_in_mongo_collection_early_stop(estimator, X, y, cv,
                                                params, scorer=scorer))
        if entry is not None:
            scores_ = np.array([entry['scores']['train_scores'],
                                entry['scores']['test_scores']]).T
            # params_copy = params.copy()
            # params_copy.update(entry['custom_params'])
            # from kaggle_tools.utils.pipeline_utils import find_xgbmodel_param_prefix
            # params_copy[find_xgbmodel_param_prefix(estimator) + 'n_estimators'] =
            current_cv_result = CVResult(estimator, X, y, cv, entry['custom_params'],
                                         scores=scores_,
                                         scorer=self.scorer_)
            return current_cv_result
        return None
