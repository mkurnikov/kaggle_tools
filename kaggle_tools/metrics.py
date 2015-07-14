from __future__ import division, print_function

import numpy as np
from sklearn.utils.validation import check_consistent_length


def rmlse(predictions, actual):
    check_consistent_length(predictions, actual)

    predictions[predictions < 0] = 0
    log_differences_squared = (np.log(predictions + 1) - np.log(actual + 1)) ** 2

    return np.sqrt(log_differences_squared.mean())
