from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np


def rmlse(predictions, actual):
    predictions[predictions < 0] = 0
    log_differences_squared = (np.log(predictions + 1) - np.log(actual + 1)) ** 2
    # print log_differences_squared
    # return log_differences_squared.sum()
    return np.sqrt(log_differences_squared.mean())