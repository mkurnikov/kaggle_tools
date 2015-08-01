from __future__ import division, print_function
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.utils.validation import _num_samples
from sklearn.cross_validation import check_scoring
from sklearn.externals.joblib import logger
import numpy as np

# from kaggle_tools.base import is_classifier, clone
from kaggle_tools.grid_search import MyGridSearchCV

from sklearn.utils.validation import check_random_state
from kaggle_tools.utils import pipeline as pipelines
import xgboost as xgb
import tools_logging


from sklearn.cross_validation import _safe_split, _index_param_value, FitFailedWarning, \
    _score, train_test_split
import time
import numbers
import warnings

def _xgb_custom_fit_and_score(estimator, X, y, scorer, train, test, verbose,
                       parameters, fit_params, return_train_score=False,
                       return_parameters=False, error_score='raise'):
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    if fit_params is None:
        raise ValueError('fit_params must have early_stopping_rounds and verbose params.')

    if 'hold_out_size' not in fit_params:
        raise ValueError('Cannot find hold_out_size param in fit_params.')

    # In order to not accidentally remove hold_out_size param.
    fit_params = fit_params.copy()

    hold_out_size = fit_params['hold_out_size']
    del fit_params['hold_out_size']
    # print(fit_params)

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if 'rng' not in fit_params:
        raise ValueError('rng param must have been passed to _xgb_custom_fit_and_score.')
    rng = fit_params['rng']
    del fit_params['rng']
    X_train, X_hold_out, y_train, y_hold_out = train_test_split(X_train, y_train,
                                                                test_size=hold_out_size, random_state=rng)

    # print(tools_logging._get_array_hash(X_hold_out), tools_logging._get_array_hash(y_hold_out))
    eval_set = [(X_hold_out, y_hold_out)]
    for key in fit_params:
        if 'eval_set' in key:
            fit_params[key] = eval_set
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
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
        # best_iteration = estimator.best_iteration
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
    parameters.update({'n_estimators': estimator.best_iteration})
    if return_parameters:
        ret.append(parameters)
    return ret


class XGBEarlyStopGridSearchCV(MyGridSearchCV):
    """Grid search class that had been implemented specifically for performing grid search on xgb model
    with early stopping. Mechanics such that on every model training early_stopping is enabled.
     Thus model can grow as long as it has to, n_estimators is being determined on training phase.

    """
    def __init__(self, estimator, param_grid, scoring=None,
             n_jobs=1, iid=False, cv=None, verbose=0,
             pre_dispatch='2*n_jobs', error_score='raise', logger=None, mongo_collection=None,
                 early_stopping_rounds=5, hold_out_size=0.05, eval_metric=None, random_state=None):

        xgb_param_prefix = pipelines.find_xgbmodel_param_prefix(estimator)[0]
        self.rng = check_random_state(random_state)
        # self.early_stopping_rounds = early_stopping_rounds
        # self.hold_out_size = hold_out_size
        fit_params = {xgb_param_prefix + 'early_stopping_rounds': early_stopping_rounds,
                          xgb_param_prefix + 'eval_metric': eval_metric,
                      xgb_param_prefix + 'verbose': False,
                      xgb_param_prefix + 'eval_set': None,
                      'hold_out_size': hold_out_size,
                      'rng': self.rng}

        super(XGBEarlyStopGridSearchCV, self).__init__(estimator, param_grid, scoring=scoring, fit_params=fit_params,
                                              n_jobs=n_jobs, iid=iid, cv=cv, verbose=verbose,
                                              pre_dispatch=pre_dispatch, error_score=error_score, logger=logger,
                                              mongo_collection=mongo_collection)
        final_ = pipelines.get_final_estimator(estimator)
        if not isinstance(final_, xgb.XGBModel):
            raise ValueError('{} can be used only for xgboost models. Current estimator class is {}'
                             .format(self.__class__.__name__, final_.__class__.__name__))


    def _make_scorer(self, estimator, scoring):
        return check_scoring(self.estimator, scoring=self.scoring)
        # check_scoring(self.estimator, scoring=self.scoring)
        # def scorer(estimator, X, y):
        #     preds = estimator.predict(X)
        #     score = self.scoring(y_test, preds)
        #     return score
        # return scorer



    def _get_custom_fit_and_score(self):
        return _xgb_custom_fit_and_score


    def _process_parameters(self, params_arr):
        processed_params = params_arr[0].copy()
        n_ests_key = None
        for key in params_arr[0].keys():
            if 'n_estimators' in key:
                n_ests_key = key
        n_estimators = np.mean([params[n_ests_key] for params in params_arr])
        processed_params.update({n_ests_key: int(n_estimators)})
        return processed_params
        # keys = params.arr[0].keys()
        # for key in keys:



if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import KFold

    X, y = make_regression(random_state=2, n_samples=1000)

    clf = xgb.XGBRegressor(nthread=1, n_estimators=1000, seed=99)

    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=22)
    fit_params = {'early_stopping_rounds': 5,
                  'eval_metric': None,
                    'verbose': True,
                  'eval_set': [(X_test, y_test)]}
    cv = KFold(len(y_train), n_folds=5, random_state=1)
    params = {
        'max_depth': [3]
    }
    search = XGBEarlyStopGridSearchCV(clf, params, cv=cv, verbose=3,
                                      random_state=10)
    search.fit(X_train, y_train)

















