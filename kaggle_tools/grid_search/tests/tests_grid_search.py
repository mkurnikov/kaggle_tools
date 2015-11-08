from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
	# noinspection PyUnresolvedReferences
	from py3compatibility import *


from kaggle_tools.grid_search._grid_search import MyGridSearchCV


def main():
    from sklearn.datasets import make_regression
    from sklearn.cross_validation import KFold

    X, y = make_regression(random_state=2)
    cv = KFold(len(y), n_folds=5, random_state=1)

    from pymongo import MongoClient
    client = MongoClient()

    collection = client['test']['grid_search']

    from sklearn.linear_model import Ridge
    clf = Ridge()
    from kaggle_tools.utils.mongo_utils import MongoSerializer, MongoCollectionWrapper
    serializer = MongoSerializer()
    collection_wrapper = MongoCollectionWrapper(serializer, collection)

    grid_search = MyGridSearchCV(clf, {'alpha': [0.1, 0.01, 0.001]}, cv=cv,
                                 verbose=3)
    grid_search.fit(X, y)

    print(grid_search.grid_scores_)


if __name__ == '__main__':
    main()