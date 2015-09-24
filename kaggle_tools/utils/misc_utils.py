from __future__ import division, print_function, \
    unicode_literals, absolute_import
# noinspection PyUnresolvedReferences
from py3compatibility import *

from sklearn.utils import murmurhash3_32


def _get_array_hash(arr):
    if hasattr(arr, 'values'):
        arr = arr.values
    return murmurhash3_32(arr.tostring())



def _get_pprinted_mean(mean):
    return '{:0.8f}'.format(mean)


def _get_pprinted_std(std):
    return '(+/-{:0.05f})'.format(std)


def _get_pprinted_cross_val_scores(scores):
    if type(scores) == list:
        scores = np.array(scores)
    msg = '{mean} {std}'.format(mean=_get_pprinted_mean(scores.mean()),
                                std=_get_pprinted_std(scores.std()))
    return msg


def pprint_cross_val_scores(scores):
    print(_get_pprinted_cross_val_scores(scores))