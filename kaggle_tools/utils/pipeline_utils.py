from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

import xgboost as xgb
from sklearn.pipeline import Pipeline


def get_final_estimator(pipeline):
    if hasattr(pipeline, '_final_estimator'):
        return get_final_estimator(pipeline._final_estimator)
    else:
        return pipeline


def find_xgbmodel_param_prefix(estimator, s=''):
    if isinstance(estimator, xgb.XGBModel):
        return '', True

    # print('s:', s)
    if isinstance(estimator, Pipeline):
        # final = estimator._final_estimator
        steps = estimator.steps
        for step in steps:
            name, est = step
            # if isinstance(est, type(final.__class__)):
            #     return '', True
            # if isinstance(est, xgb.XGBModel):
            #     return '__'.join([name, s]), True

            s, is_cont = find_xgbmodel_param_prefix(est, s)
            if is_cont:
                return '__'.join([name, s]), True
            else:
                continue

    return s, False