from __future__ import division, print_function

import numpy as np
from sklearn.utils.validation import check_consistent_length


def rmlse(predictions, actual):
    check_consistent_length(predictions, actual)

    predictions[predictions < 0] = 0
    log_differences_squared = (np.log(predictions + 1) - np.log(actual + 1)) ** 2

    return np.sqrt(log_differences_squared.mean())

import xgboost as xgb

def xgb_normalized_gini(y_pred, y_true):
    if isinstance(y_pred, xgb.DMatrix):
        y_pred = y_pred.get_label()

    if isinstance(y_true, xgb.DMatrix):
        y_true = y_true.get_label()

    # print(y_pred.shape, y_true.shape)
    return 'gini', 1 - normalized_gini(y_true, y_pred)
    # print(y_pred)
    # print(y_true)
    # raise SystemExit(1)

def normalized_gini(y_true, y_pred):
    # check and get number of samples
    #
    # y_true **= 2

    # if isinstance(y_pred, xgb.DMatrix):
    #     print(y_pred.__dict__)
    #     y_pred = y_pred.data
    y_pred = y_pred.reshape(y_true.shape)
    # print(y_true.shape, y_pred.shape)
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred / G_true
