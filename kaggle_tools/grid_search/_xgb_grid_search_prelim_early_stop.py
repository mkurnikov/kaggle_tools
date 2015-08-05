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

from kaggle_tools.grid_search import XGBEarlyStopGridSearchCV


class XGBOneEarlyStopThenCV_GridSearchCV(XGBEarlyStopGridSearchCV):
    def _fit_and_cv_result(self, estimator, X, y, parameters, cv,
                           parallel=None, scorer=None, verbose=0):
        if self.collection_wrapper is not None:
            existing_cv_result = self._extract_existing_cv_result(estimator, X, y, cv,
                                                                  parameters, scorer=self.scorer_)
            if existing_cv_result is not None:
                return existing_cv_result

        # steps = dict(estimator.steps)
        # union = steps['features']
        # estimator = steps['estimator']

        X_train, X_hold_out, y_train, y_hold_out = train_test_split(X, y,
                                                            test_size=self.hold_out_size,
                                                            random_state=self.rng)

        # X_train = X.ix[X_train_indx]
        # X_train = union.fit_transform(X_train)
        # X_hold_out = union.fit_transform(X.ix[X_hold_out_indx])
        # if isinstance(estimator, Pipeline):


        # estimator.steps

        estimator_prelim = clone(estimator)
        eval_set = [(X_train, y_train), (X_hold_out, y_hold_out)]
        xgb_param_prefix = pipeline_utils.find_xgbmodel_param_prefix(estimator_prelim)[0]
        fit_params = {
            xgb_param_prefix + 'eval_set': eval_set,
            xgb_param_prefix + 'early_stopping_rounds': self.early_stopping_rounds,
            xgb_param_prefix + 'eval_metric': self.eval_metric,
            xgb_param_prefix + 'verbose': self.verbose_eval,
            xgb_param_prefix + 'maximize_score': self.maximize_score
        }
        estimator_prelim.fit(X_train, y_train, **fit_params)
        n_ests_dict = {
            str(xgb_param_prefix + 'n_estimators'): pipeline_utils.get_final_estimator(estimator_prelim).best_iteration
        }
        parameters.update(n_ests_dict)

        # estimator_prelim = clone(estimator)
        # eval_set = [(X_train, y_train), (X_hold_out, y_hold_out)]
        # xgb_param_prefix = pipeline_utils.find_xgbmodel_param_prefix(estimator_prelim)[0]
        # fit_params = {
        #     xgb_param_prefix + 'eval_set': eval_set,
        #     xgb_param_prefix + 'early_stopping_rounds': self.early_stopping_rounds,
        #     xgb_param_prefix + 'eval_metric': self.eval_metric,
        #     xgb_param_prefix + 'verbose': self.verbose_eval,
        #     xgb_param_prefix + 'maximize_score': self.maximize_score
        # }
        # estimator_prelim.fit(X_train, y_train, **fit_params)
        # n_ests_dict = {
        #     str(xgb_param_prefix + 'n_estimators'): pipeline_utils.get_final_estimator(estimator_prelim).best_iteration
        # }
        # parameters.update(n_ests_dict)
        # print(parameters)

        # out = parallel(
        #     delayed(xgb_early_stop_fit_and_score)(clone(estimator), X, y, scorer,
        #                                    train, test, verbose, parameters,
        #                                    return_train_score=True, error_score=self.error_score,
        #                                    hold_out_size=self.hold_out_size, early_stopping_rounds=self.early_stopping_rounds,
        #                                    verbose_eval=self.verbose_eval, eval_metric=self.eval_metric,
        #                                           random_state=self.rng, maximize_score=self.maximize_score)
        #     for train, test in cv)
        out = parallel(
            delayed(custom_fit_and_score_)(clone(estimator), X, y, scorer,
                                    train, test, verbose, parameters,
                                    return_train_score=True, error_score=self.error_score)
                for train, test in cv)

        print(parameters)
        scores = np.array(out)[:, [0, 1]]
        # params_arr = np.array(out)[:, [4]].flatten()
        # #
        # processed_params = params_arr[0].copy() #example of keys
        # n_ests_key = None
        # for key in processed_params:
        #     if 'n_estimators' in key:
        #         n_ests_key = key
        # n_estimators = np.mean([params[n_ests_key] for params in params_arr.flatten()])
        # processed_params.update({n_ests_key: int(n_estimators)})


        current_cv_result = CVResult(estimator, X, y, cv, parameters,
                                     scores=scores, scorer=self.scorer_)
        if self.logger is not None:
            self.logger.info(current_cv_result)
        if self.collection_wrapper is not None:
            self.collection_wrapper.insert_cv_result(current_cv_result)

        return current_cv_result

