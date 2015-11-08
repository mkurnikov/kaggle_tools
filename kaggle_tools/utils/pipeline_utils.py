from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
	# noinspection PyUnresolvedReferences
	from py3compatibility import *


try:
    import xgboost as xgb
except Exception as e:
    pass

from sklearn.pipeline import Pipeline


def get_final_estimator(pipeline):
    if hasattr(pipeline, '_final_estimator'):
        return get_final_estimator(pipeline._final_estimator)
    else:
        return pipeline

from sklearn.base import ClassifierMixin, RegressorMixin

def find_final_estimator_param_prefix(estimator, s=''):
    if isinstance(estimator, (ClassifierMixin, RegressorMixin)):
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

            s, is_cont = find_final_estimator_param_prefix(est, s)
            if is_cont:
                return '__'.join([name, s]), True
            else:
                continue

    return s, False