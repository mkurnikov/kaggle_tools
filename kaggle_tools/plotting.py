from __future__ import division, print_function

from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def show_feature_importances(estimator, feature_names=None):
    """
        Show bar graph for feature importances for estimators like RandomForest and GradientBoosting
    """
    if not hasattr(estimator, 'feature_importances_'):
        raise AttributeError('Estimator {} doesn\'t support feature importances'.format(estimator))

    importances = estimator.feature_importances_
    feature_importance = 100.0 * (importances / importances.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.legend(loc='best')
    return plt


def pprint_grid_scores(grid_scores, sorted_by_mean_score=True):
    if sorted_by_mean_score:
        grid_scores = sorted(grid_scores, key=lambda x: x.mean_validation_score)
    for cv_scores in grid_scores:
        params_, mean_score, scores, _, _ = cv_scores
        print('%0.8f (+/-%0.05f) for %r'
              % (mean_score, scores.std(), params_))


def plot_train_test_error(param_name, param_grid, grid_search_,
                          more_is_better=False, show_train_error=True,
                          print_grid_scores=False):
    train_errors = []
    train_error_stds = []
    test_errors = []
    test_error_stds = []
    for cv_scores in grid_search_.grid_scores_:
        # print(cv_scores)
        params_, mean_score, scores, train_score, train_scores = cv_scores
        # print scores
        # estimators_pipeline.set_params(**params_)
        # estimators_pipeline.fit(dataset, target)
        # train_error = 1 - accuracy_score(target, estimators_pipeline.predict(dataset))
        # print('train error:', 1 - train_score)
        if not more_is_better:
            train_errors.append(1 - train_score)
            train_error_stds.append(train_scores.std() / 2)

            test_errors.append(1 - mean_score)
            test_error_stds.append(scores.std() / 2)
        else:
            train_errors.append(train_score)
            train_error_stds.append(train_scores.std() / 2)

            test_errors.append(mean_score)
            test_error_stds.append(scores.std() / 2)

    if print_grid_scores:
        pprint_grid_scores(grid_search_.grid_scores_)

    errors = pd.DataFrame(data={
        # 'C': params['forest__n_estimators'],
        'param': param_grid[param_name],
        # 'param': params['forest__max_features'],
        # 'param' : params['boosting__n_estimators'],
        # 'param' : params['boosting__max_depth'],
        # 'param' : params['boosting__learning_rate'],
        'train_error': train_errors,
        'train_error_std': train_error_stds,
        'test_error': test_errors,
        'test_error_std': test_error_stds
    })

    fig = plt.figure()
    if show_train_error:
        plt.plot(errors['param'], errors['train_error'], 'k-', label='train_error', linewidth=2)
        plt.plot(errors['param'], errors['train_error'], 'k.', markersize=12)

    plt.plot(errors['param'], errors['test_error'], 'r-', label='test_error', linewidth=2)

    ax = fig.gca()  # get current axis
    ax.errorbar(errors['param'], errors['test_error'], fmt='r.', markersize=12, yerr=errors['test_error_std'])

    plt.plot(errors['param'], errors['test_error'], 'r.', markersize=12)
    plt.xlabel(param_name)
    plt.ylabel('error')
    # plt.xlim(1, 12)
    plt.title('Train/test error plots')
    plt.legend(loc='best')

    # best_mean, best_std = grid_search_.best_.mean_validation_score, np.std(grid_search_.best_.cv_validation_scores) / 2
    # bottom = 1 - (best_mean + best_std)
    # height = best_std * 2
    # plt.barh(bottom, ax.xmax, height=height, color='red', alpha=0.2)
    return fig, ax
