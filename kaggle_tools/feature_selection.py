from __future__ import division, absolute_import, print_function, unicode_literals

import itertools
import six
import collections
from abc import ABCMeta, abstractmethod
import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import cross_val_score
from sklearn.utils.validation import check_X_y, NotFittedError


# TODO - implement sebastian rachka ipython notebook here

class BaseFSCV(six.with_metaclass(ABCMeta, BaseEstimator)):
    """
        Base class for all feature selection algorithms.
        Includes cross_val_score algorithm rewritten to do CV on selected subset.

        Best subset always corresponds to maximum of evaluation criteria.
        Please, write "scorer" parameter correctly.

        "scorer" signature: scorer(estimator, X, y)
    """

    # TODO: add custom cross-validation scoring ???
    # TODO - add mapping between feature and its name
    # TODO - add logging
    # TODO - add verbose prints
    def __init__(self, estimator, scorer, cv=None, logging_fname=None, n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.scorer = scorer
        self.logging_fname = logging_fname
        self.n_jobs = n_jobs
        self.feature_mask = None


    @abstractmethod
    def is_mask_incomplete(self):
        raise NotImplementedError


    @abstractmethod
    def get_augmented_feature_mask(self, feature_set):
        raise NotImplementedError


    def _cross_val_score_on_fsubset(self, X, y, subset):
        base_estimator = clone(self.estimator)
        return cross_val_score(base_estimator, X[:, subset], y, scoring=self.scorer,
                               cv=self.cv, n_jobs=self.n_jobs)


    def init_feature_mask(self, n_features, is_forward=True):
        if is_forward:
            self.feature_mask = np.zeros((n_features,), dtype=np.bool)
        else:
            self.feature_mask = np.ones((n_features,), dtype=np.bool)


    def add_features(self, feature_indices, create_new=True):
        if create_new:
            augmented_mask = np.copy(self.feature_mask)
            augmented_mask[feature_indices] = True
            return augmented_mask
        else:
            self.feature_mask[feature_indices] = True


    def remove_features(self, feature_indices, create_new=True):
        if create_new:
            augmented_mask = np.copy(self.feature_mask)
            augmented_mask[feature_indices] = False
            return augmented_mask
        else:
            self.feature_mask[feature_indices] = False


    def get_feature_candidates(self, is_forward=True):
        if is_forward:
            # False value - correct
            return np.where(~self.feature_mask)[0]
        else:
            # True value - correct
            return np.where(self.feature_mask)[0]


    def get_feature_mask(self):
        if self.feature_mask is None:
            raise NotFittedError
        return self.feature_mask



class SequentialForwardFSCV(BaseFSCV):
    """
        Or greedy forward feature selection.
        Starts from an empty set of features and on every step adds one best feature to it.
    """

    # TODO: add early stopping and documentation for it
    def __init__(self, estimator, scorer, cv=None, logging_fname=None, n_jobs=1,
                 minimal_mean_diff=0.0, expected_n_features=-1):
        super(SequentialForwardFSCV, self).__init__(estimator,
                                                    scorer,
                                                    cv=cv,
                                                    logging_fname=logging_fname,
                                                    n_jobs=n_jobs)
        # self.more_better = more_better
        self.minimal_diff = minimal_mean_diff
        if expected_n_features != -1:
            raise NotImplementedError('Specification of desired feature subset size is not ready yet.')

        # TODO - add support for desired fsubset size selection
        if expected_n_features == -1:
            self.expected_n_features = np.PINF


    def is_mask_incomplete(self):
        # check for expected features here
        return not all(self.feature_mask)


    def get_augmented_feature_mask(self, feature_set):
        # if isinstance(feature_set, collections.Iterable):
        #     self.add_features(feature_set)
        return self.add_features([feature_set], create_new=True)


    def fit(self, X, y):
        X, y = check_X_y(X, y)

        n_features = X.shape[1]
        self.init_feature_mask(n_features, is_forward=True)

        previous_score = 0
        iteration = 0
        while self.is_mask_incomplete():
            iteration += 1
            scores_list = []

            for feature_candidate in self.get_feature_candidates(is_forward=True):
                augmented_mask = self.get_augmented_feature_mask(feature_candidate)
                scores = self._cross_val_score_on_fsubset(X, y, augmented_mask)

                scores_list.append((feature_candidate, scores))

            # choose best feature on this iteration
            best_feature, best_scores = max(scores_list, key=lambda x: x[1].mean())
            score_diff = best_scores.mean() - previous_score

            # stop, if we didn't manage to find new feature with appropriate performance gain
            if score_diff < self.minimal_diff:
                break

            print('{}. Best score: {:.6f} +/- {:.6f}'.format(iteration, best_scores.mean(), best_scores.std()))

            self.add_features([best_feature], create_new=False)
            previous_score = best_scores.mean()
        return self.get_feature_mask()



# TODO - add support for best three features search(like in article), maybe new mixin and different hierarchy.
# it will allow testing of algoritms
class SequentialFeatureSelectionMixin(object):
    # TODO - implement
    pass

# i've only changed add_features -> remove_feature, is_incomplete and is_forward. something's wrong.
class SequentialBackwardFECV(BaseFSCV):
    """
        Backward feature elimination algorithm.
    """
    # TODO: test it
    # TODO: add reference documentation
    def __init__(self, estimator, scorer, cv=None, logging_fname=None, n_jobs=1,
                 more_better=True, minimal_mean_diff=0.0):
        super(SequentialBackwardFECV, self).__init__(estimator,
                                                     scorer,
                                                     cv=cv,
                                                     logging_fname=logging_fname,
                                                     n_jobs=n_jobs)
        self.minimal_diff = minimal_mean_diff


    def is_mask_incomplete(self):
        # check for expected features here
        return any(self.feature_mask)


    def get_augmented_feature_mask(self, feature_set):
        # if isinstance(feature_set, collections.Iterable):
        #     self.add_features(feature_set)
        return self.remove_features([feature_set], create_new=True)


    def fit(self, X, y):
        X, y = check_X_y(X, y)

        n_features = X.shape[1]
        self.init_feature_mask(n_features, is_forward=False)

        previous_score = 0
        iteration = 0
        while self.is_mask_incomplete():
            iteration += 1
            scores_list = []

            for feature_candidate in self.get_feature_candidates(is_forward=False):
                augmented_mask = self.get_augmented_feature_mask(feature_candidate)
                scores = self._cross_val_score_on_fsubset(X, y, augmented_mask)

                scores_list.append((feature_candidate, scores))

            # choose best feature on this iteration
            best_feature, best_scores = max(scores_list, key=lambda x: x[1].mean())
            score_diff = best_scores.mean() - previous_score

            # stop, if we didn't manage to find new feature with appropriate performance gain
            if score_diff < self.minimal_diff:
                break

            print('{}. Best score: {:.6f} +/- {:.6f}'.format(iteration, best_scores.mean(), best_scores.std()))

            self.remove_features([best_feature], create_new=False)
            previous_score = best_scores.mean()
        return self.get_feature_mask()


class FullSearchFSCV(BaseFSCV):
    """
        Search over all possible combinations of features. Extremely slow, use only where n_features is about 10.
    """

    # TODO: replace all prints with logging, or maybe with generator(optional, similar to the RandomForest)
    # TODO: more documentation
    # TODO: rewrite to general form
    def __init__(self, estimator, scorer, cv=None, logging_fname=None, n_jobs=1):
        super(FullSearchFSCV, self).__init__(estimator,
                                             scorer,
                                             cv=cv,
                                             # scoring=scoring,
                                             logging_fname=logging_fname,
                                             n_jobs=n_jobs)

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # feature_mask = np.zeros((X.shape[1],), dtype=np.bool)
        n_features = X.shape[1]
        features = range(0, n_features)

        n_features = 7
        subsets_completed = 0
        # all possible combinations of features
        feature_combinations_score = []
        # CombinationScore = namedtuple('CombinationScore', ['combination', 'mean', 'std'])
        for L in range(6, n_features + 1):
            # print('for feature array length:', L)
            for subset in itertools.combinations(features, L):
                # print('\n', subset)
                # print subset
                scores = self._cross_val_score_on_fsubset(X, y, subset)
                subsets_completed += 1
                if subsets_completed % 1000 == 0:
                    print('subsets completed: {}'.format(subsets_completed))
                feature_combinations_score.append(
                    (subset, scores.mean(), scores.std()))
        # rewrite into tuple(list, list)
        # print('features comb len: ', len(feature_combinations_score))
        for subset, score_mean, score_std in sorted(feature_combinations_score,
                                                    key=lambda cscore: cscore[1], reverse=True)[:100]:
            print('{:.6f} +/- {:.6f} {}'.format(score_mean, score_std, subset))


class FullSearchFSCVGridSearch(FullSearchFSCV):
    def _cross_val_score_on_fsubset(self, X, y, subset):
        # self.estimator.fit(X, y)
        grid_search = clone(self.estimator)
        grid_search.fit(X[:, subset], y)
        # print grid_search.best_estimator_
        print(grid_search.best_score_, grid_search.best_params_)
        # base_estimator = clone(self.estimator)
        # return grid_search.best_score_
        return cross_val_score(grid_search.best_estimator_, X[:, subset], y, scoring=self.scoring,
                               cv=self.cv, n_jobs=self.n_jobs)
