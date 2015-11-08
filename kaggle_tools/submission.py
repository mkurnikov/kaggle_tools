from __future__ import division, print_function, \
    unicode_literals, absolute_import

import six
if six.PY2:
	# noinspection PyUnresolvedReferences
	from py3compatibility import *


from abc import ABCMeta, abstractmethod, abstractproperty
import pickle
import os
from sklearn.externals import six
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import check_cv
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

from kaggle_tools.cross_validation import my_cross_val_score
from kaggle_tools.utils.misc_utils import pprint_cross_val_scores
from kaggle_tools.utils.mongo_utils import MongoSerializer, MongoCollectionWrapper


class BaseSubmittion(six.with_metaclass(ABCMeta)):
    """
    """
    def __init__(self, pipeline, submission_id,
                 cv=None, early_stopping=False):
        # self.train_set = train_set
        # self.test_set = test_set
        self.pipeline = pipeline
        self.submission_id = submission_id
        if cv is not None:
            cv = check_cv(cv)
        self.cv = cv
        self.early_stopping = early_stopping
        # self.serialize_fitted_pipeline = serialize_fitted_pipeline

        self.cv_scores = None
        self.submission_score = None
        self.is_fitted = False
        self.original_test_set = None


    def fit(self, X, y, perform_cv=True, scoring=None, n_jobs=1, verbose=0):
        if perform_cv:
            if scoring is None:
                raise ValueError('scoring parameter can not be None with perform_cv=True. '
                                 'Scoring must be presented to perform CV.')
            self.perform_cv(X, y, scoring, n_jobs=n_jobs, verbose=verbose)

        print('fitting estimator with full data...')
        import time
        before = time.time()

        self.pipeline.fit(X, y)
        self.is_fitted = True
        print('full data fitted. time:', time.time() - before)


    def predict(self, X):
        """Creating submission file out of already trained pipeline.
        """
        if not self.is_fitted:
            raise ValueError('Pipeline is not fitted.')

        self.original_test_set = X
        predictions = self.pipeline.predict(X).ravel()
        return predictions


    @abstractmethod
    def create_submission(self, predictions, original_test_set, submission_file):
        """ Filesystem logic to save submission to file.
        """
        pass


    # @abstractproperty
    # def project_submission_id_(self):
    #     """ID for specific type of submissions. Like __class__.__name__.
    #     In case of different types of base submissions in project.
    #     """
    #     pass
    #
    #
    # @abstractproperty
    # def serialized_models_directory_(self):
    #     """Directory for pickled models.
    #     """
    #     pass
    #
    #
    # @abstractproperty
    # def submission_mongo_collection_(self):
    #     """Specify which collection should be used to save results.
    #     """
    #     pass


    def _save(self, show_json=False, serialize_fitted_pipeline=False):
        if not self.is_fitted:
            raise ValueError('Pipeline is not fitted.')

        self.submission_score = input('Enter submission score: ')

        if serialize_fitted_pipeline:
            dest_path = os.path.join(self.serialized_models_directory_, self.submission_id) + '.pkl'
            dest = open(dest_path, 'wb')
            pickle.dump(self.pipeline, dest)

            print('Model has been pickled to {}'.format(dest_path))

        serializer = MongoSerializer()
        if show_json:
            import json
            json_obj = serializer.serialize(self)
            print(json.dumps(json_obj, indent=2, sort_keys=True))

        collection_wrapper = MongoCollectionWrapper(serializer, self.submission_mongo_collection_)
        collection_wrapper.insert_submission(self)


    def perform_cv(self, X, y, scoring, n_jobs=1, verbose=0):
        pipeline = clone(self.pipeline)
        scores = my_cross_val_score(pipeline, X, y,
                           verbose=verbose, scoring=scoring, cv=self.cv, n_jobs=n_jobs)
        print('Cross validation scores for the training set:', end=' ')
        pprint_cross_val_scores(scores)

        self.cv_scores = scores


    @property
    def json_(self):
        return MongoSerializer()._submission(self)

















