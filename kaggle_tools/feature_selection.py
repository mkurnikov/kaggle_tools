from __future__ import division, absolute_import, print_function, unicode_literals

import itertools

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.cross_validation import cross_val_score


class BaseFSCV(BaseEstimator):
    # TODO: add documentation
    # TODO: add custom cross-validation scoring
    def __init__(self, estimator, cv=None, scoring=None, logging_fname=None, n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.scoring = scoring
        self.logging_fname = logging_fname
        self.n_jobs = n_jobs

    def _cross_val_score_on_fsubset(self, X, y, subset):
        base_estimator = clone(self.estimator)
        return cross_val_score(base_estimator, X[:, subset], y, scoring=self.scoring,
                               cv=self.cv, n_jobs=self.n_jobs)


class GreedyForwardFSCV(BaseFSCV):
    # TODO: all documentation
    # TODO: add early stopping
    def __init__(self, estimator, cv=None, scoring=None, logging_fname=None, n_jobs=1,
                 more_better=True, minimal_mean_diff=0.0):
        super(GreedyForwardFSCV, self).__init__(estimator,
                                                cv=cv,
                                                scoring=scoring,
                                                logging_fname=logging_fname,
                                                n_jobs=n_jobs)
        self.more_better = more_better
        self.minimal_diff = minimal_mean_diff


    def fit(self, X, y):
        n_features = X.shape[1]
        feature_mask = np.zeros((n_features,), dtype=np.bool)
        if self.more_better:
            previous_score = 0.0
        else:
            previous_score = 9999.0

        chosen_features = []
        # while we don't go through every "level" of features, repeat
        iteration = 0
        # previous_std = 9999.9
        while not all(feature_mask):
            iteration += 1
            scores_list = []
            features_to_choose_from = np.where(~feature_mask)[0]
            for feature_to_choose_idx in features_to_choose_from:
                augmented_mask = np.copy(feature_mask)
                augmented_mask[feature_to_choose_idx] = True
                scores = self._cross_val_score_on_fsubset(X, y, augmented_mask)
                scores_list.append((feature_to_choose_idx, scores))

            # choose best feature on this iteration
            if self.more_better:
                best_feature, best_scores = max(scores_list, key=lambda x: x[1].mean())
                score_diff = best_scores.mean() - previous_score
            else:
                best_feature, best_scores = min(scores_list, key=lambda x: x[1].mean())
                score_diff = previous_score - best_scores.mean()

            if score_diff < self.minimal_diff:
                break
                # if score_diff > -0.01 and best_scores.std() < previous_std - 0.01:
                # pass
                # else:
                # break

            chosen_features.append(best_feature)
            print('{}. Best score: {:.6f} +/- {:.6f}'.format(iteration, best_scores.mean(), best_scores.std()))

            feature_mask[best_feature] = True
            previous_score = best_scores.mean()
            # previous_std = best_scores.std()
        return feature_mask


class BackwardFECV(BaseFSCV):
    """
        Backward feature elimination algorithm.
    """
    # TODO: implement it
    # TODO: add reference documentation
    def __init__(self, estimator, cv=None, scoring=None, logging_fname=None, n_jobs=1,
                 more_better=True, minimal_mean_diff=0.0):
        super(BackwardFECV, self).__init__(estimator,
                                           cv=cv,
                                           scoring=scoring,
                                           logging_fname=logging_fname,
                                           n_jobs=n_jobs)
        self.more_better = more_better
        self.minimal_diff = minimal_mean_diff


    def fit(self, X, y):
        raise NotImplementedError


class FullSearchFSCV(BaseFSCV):
    """
        Search over all possible combinations of features. Extremely slow, use only where n_features is about 10.
    """

    # TODO: replace all prints with logging, or maybe with generator(optional, similar to the RandomForest)
    # TODO: more documentation
    def __init__(self, estimator, cv=None, scoring=None, logging_fname=None, n_jobs=1):
        super(FullSearchFSCV, self).__init__(estimator,
                                             cv=cv,
                                             scoring=scoring,
                                             logging_fname=logging_fname,
                                             n_jobs=n_jobs)

    def fit(self, X, y):
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