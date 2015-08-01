from __future__ import division, print_function

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.extmath import cartesian
from sklearn.utils.validation import check_X_y, check_array


class BaseEnsembleEstimator(BaseEstimator):
    # TODO - move all general code here
    pass


class EnsembleClassifier(BaseEnsembleEstimator, ClassifierMixin):
    #VotingClassifier satisfy.

    # TODO - majorite vote for prediction
    # TODO - calibration
    # TODO - average(or some other metric) for predict_proba
    pass

import warnings
class EnsembleRegressor(BaseEnsembleEstimator, RegressorMixin):
    """
        Meta estimator that averages predictions from all level-0 estimators.
    """
    def __init__(self, estimators=None, weights=None, subsample=1.0, prediction_transform=None):
        self.estimators = estimators
        if weights is not None and sum(weights) != 1.0:
            warnings.warn('Weights does not sum up to 1.0.')
            # raise AttributeError('Estimator weights must sum up to one. '
            #                      'Current weights = {}, sum up to {}'.format(weights, sum(weights)))
        self.weights = weights
        self.subsample = subsample
        self.preds_transform = prediction_transform


    def fit(self, X, y):
        # X, y = check_X_y(X, y)
        for estimator in self.estimators:
            if not hasattr(estimator, 'fit'):
                raise RuntimeError('Base estimator %s doesn\'t have fit() method' % str(estimator))

            estimator.fit(X, y)


    # def _get_minimization_objective(self, preds, y, scoring_func):
    #     def objective(weights):
    #         return scoring_func(np.average(preds, axis=1, weights=weights), y)
    #     return objective

    #
    # def fit_weights(self, X, y, scoring_func, step=0.05):
    #     X, y = check_X_y(X, y)
    #     n_estimators = len(self.estimators)
    #
    #     predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float32)
    #     for i, estimator in enumerate(self.estimators):
    #         predictions[:, i] = self._predict_local(estimator, X)
    #
    #     grid_line = np.arange(0.0, 1.0+step, step=step)
    #     grid = cartesian(tuple([grid_line] * n_estimators))
    #
    #     best_weights = None
    #     min_func = 1.0
    #     for weights_set in grid:
    #         # for some reason, exact match doesn't work
    #         if abs(sum(weights_set) - 1.0) > 0.00001:
    #             continue
    #
    #         objective = self._get_minimization_objective(predictions, y, scoring_func)
    #         current_func = objective(weights_set)
    #         # print(current_func)
    #         if current_func < min_func:
    #             min_func = current_func
    #             best_weights = weights_set
    #     self.weights = best_weights
    #     if self.weights is None:
    #         print('Weights fitting wasnt successful, using uniform weights instead.')
    #     return self.weights


    def _predict_local(self, estimator, X):
        if not hasattr(estimator, 'predict'):
            raise RuntimeError('Base estimator %s doesn\'t have predict() method' % str(estimator))

        prediction = estimator.predict(X)
        if self.preds_transform is not None:
            if not hasattr(self.preds_transform, '__call__'):
                raise RuntimeError('preds_transform is not callable')
            prediction = self.preds_transform(prediction)
        return prediction


    def predict(self, X):
        # X = check_array(X)

        predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float32)
        for i, estimator in enumerate(self.estimators):
            prediction = self._predict_local(estimator, X)
            if len(prediction.shape) == 2:
                prediction = prediction.reshape((prediction.shape[0],))
            predictions[:, i] = prediction

        return np.average(predictions, axis=1, weights=self.weights)


    def scorer(self, X, y, scoring_func=None, verbose=False):
        X, y = check_array(X, y)

        predictions = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float32)
        for i, estimator in enumerate(self.estimators):
            predictions[:, i] = self._predict_local(estimator, X)
            if verbose:
                print('Classifier: %s, score: %f' % (str(estimator), scoring_func(predictions[:, i], y)))
        return scoring_func(predictions.mean(axis=1), y)


if __name__ == '__main__':
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import train_test_split

    X, y = make_regression(random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf1 = Ridge()
    clf2 = Lasso(random_state=1)
    clf3 = Lasso(alpha=16.0)

    ensemble = EnsembleRegressor(estimators=[clf1, clf2, clf3])
    ensemble.fit(X_train, y_train)
    print(ensemble.score(X_test, y_test))

    from sklearn.metrics import r2_score
    def obj(*args, **kwargs):
        return 1 - r2_score(*args, **kwargs)
    print(ensemble.fit_weights(X_test, y_test, obj, step=0.05))



