from __future__ import division, absolute_import, print_function, unicode_literals

import numpy as np
from scipy import signal
from sklearn.utils.extmath import cartesian
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y


class WindowOffsetsAmplitudesGenerator(BaseEstimator, TransformerMixin):
    """
        Finds average amplitudes of a signal obtained at different offsets from the start and at different window sizes
            piece = offset : offset + window_size

            allow_partial_slices - if want to allow cropped(incomplete) windows
    """
    # TODO - possibly incorrect, tested only for runnability
    def __init__(self, window_sizes=None, offsets=None, allow_partial_slices=False):
        if window_sizes is None:
            raise AttributeError('Window sizes array is not specified.')
        if offsets is None:
            raise AttributeError('Offsets array is not specified.')

        self.offsets = offsets
        self.window_sizes = window_sizes
        self.allow_partial_slices = allow_partial_slices


    def fit_transform(self, X, y=None, **fit_params):
        """
            X shape should be (n_samples, n_time_ticks). Output shape is (n_samples, len(offsets) * len(window_sizes)
                that is satisfy to the offset + window_size < end_of_response)
        """
        n_samples, n_ticks = X.shape

        offset_window_combs = cartesian((self.offsets, self.window_sizes))
        if not self.allow_partial_slices:
            offset_window_combs = offset_window_combs[np.sum(offset_window_combs, axis=1) < n_ticks]

        amplitudes = np.zeros((n_samples, len(offset_window_combs)))
        for i, (offset, window) in enumerate(offset_window_combs):
            amplitudes[:, i] = X[:, offset:offset+window].mean(axis=1)
        return amplitudes


class TemplateCovariancesGenerator(BaseEstimator, TransformerMixin):
    """
        Finds correlations, covariances, maximum cross correlations and distance for average signal(template).
        Demands as input (n_samples, n_ticks) ndarray for one channel.
    """
    def __init__(self, boundaries=None):
        if boundaries is None:
            raise AttributeError('Boundaries of a signal cannot be None')
        self.boundaries = boundaries
        self.template_len = boundaries[1] - boundaries[0] + 1
        if self.boundaries[0] >= self.boundaries[1]:
            raise AttributeError('Left boundary should be less than right boundary.')

        # TODO - check for +1 length
        self.positive_fb_template = np.zeros((self.template_len,))
        self.negative_fb_template = np.zeros((self.template_len,))


    def fit_transform(self, X, y=None, **fit_params):
        # TODO - maybe add support for more than two values for feedback
        # TODO - add distances and cross-correlations
        # TODO - test generator
        if y is None:
            raise AttributeError('Cannot extract information about feedback.')
        X, y = check_X_y(X, y)
        self._check_boundaries_compatibility(X)

        positive_rows = (y == 1)
        negative_rows = (y == 0)
        X_positive_cropped = X[positive_rows][:, self.boundaries[0]:self.boundaries[1]+1]
        X_negative_cropped = X[negative_rows][:, self.boundaries[0]:self.boundaries[1]+1]

        self.positive_fb_template = X_positive_cropped.sum() / self.template_len
        self.negative_fb_template = X_negative_cropped.sum() / self.template_len

        correlations = np.zeros(y.shape)
        covariances = np.zeros(y.shape)
        max_xcorrelations = np.zeros(y.shape)
        distances = np.zeros(y.shape)

        # correlations
        correlations[positive_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.positive_fb_template),
                                                          axis=0, arr=X_positive_cropped)
        correlations[negative_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.positive_fb_template),
                                                          axis=0, arr=X_negative_cropped)

        # covariances
        covariances[positive_rows] = np.apply_along_axis(lambda row: np.cov(row, self.positive_fb_template)[0, 1],
                                                          axis=0, arr=X_positive_cropped)
        covariances[negative_rows] = np.apply_along_axis(lambda row: np.cov(row, self.negative_fb_template)[0, 1],
                                                          axis=0, arr=X_negative_cropped)
        #
        # # max crosscorrelations
        # max_xcorrelations[positive_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.positive_fb_template),
        #                                                   axis=0, arr=X_positive_cropped)
        # max_xcorrelations[negative_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.negative_fb_template),
        #                                                   axis=0, arr=X_negative_cropped)
        #
        # # distances
        # correlations[positive_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.positive_fb_template), axis=0,
        #                     arr=X_positive_cropped)
        # correlations[negative_rows] = np.apply_along_axis(lambda row: np.correlate(row, self.positive_fb_template), axis=0,
        #                     arr=X_negative_cropped)
    #     i can assign array[index] = arr2, where arr2.shape = index.shape


    def _check_boundaries_compatibility(self, X):
        n_ticks = X.shape[1]
        if self.boundaries[0] < 0 or self.boundaries[1] > n_ticks:
            raise AttributeError('Boundaries exceed signal borders, something is wrong.')


if __name__ == '__main__':
    X = np.array([range(100), range(100)])
    print(X.shape)

    transformer = WindowOffsetsAmplitudesGenerator(window_sizes=range(10, 100, 10),
                                                   offsets=range(10, 100, 10),
                                                   allow_partial_slices=False)
    print(transformer.fit_transform(X))



