from __future__ import division, print_function

import numpy as np
from itertools import chain

from sklearn.cross_validation import _PartitionIterator
from sklearn.utils.random import check_random_state, sample_without_replacement
from sklearn.utils import safe_indexing


MAX_SEED = 4294967295

class LeavePSubjectsOutKFold(_PartitionIterator):
    """
        Sample p subjects from subject set for each fold and creates test mask for dataset based on that subjects,
            then iterates over them.
    """
    # TODO - documentation for parameters
    def __init__(self, subjects, p, n_folds=3, random_state=None):
        super(LeavePSubjectsOutKFold, self).__init__(len(subjects), indices=None)
        self.subjects = np.array(subjects, copy=True)
        self.unique_subjects, self.subj_indices = np.unique(self.subjects, return_inverse=True)
        self.n_unique_subjects = len(self.unique_subjects)
        self.p = p
        self.n_folds = n_folds
        self.random_state = check_random_state(random_state)


    def _iter_test_masks(self):
        for fold in range(self.n_folds):
            subject_subset_idx = sample_without_replacement(n_population=self.n_unique_subjects,
                                       n_samples=self.p,
                                       random_state=self.random_state.randint(0, MAX_SEED),
                                       method='auto')
            test_mask = np.in1d(self.subj_indices, subject_subset_idx)
            yield test_mask


    def _iter_test_indices(self):
        raise NotImplementedError


def subjectwise_train_test_split(subjects, *arrays, **options):
    """
        Perform train_test_split subject-wise, based on subjects array.
    """
    # TODO - documentation for parameters

    n_test_subjects = options.pop('n_test_subjects', None)
    random_state = options.pop('random_state', None)

    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")
    if options:
        raise TypeError("Invalid parameters passed: %s" % str(options))
    if n_test_subjects is None:
        n_test_subjects = np.floor(len(subjects) * 0.25)

    cv = LeavePSubjectsOutKFold(subjects, n_test_subjects, random_state=random_state)

    train, test = next(iter(cv))
    return list(chain.from_iterable((safe_indexing(a, train),
                                         safe_indexing(a, test)) for a in arrays))





if __name__ == '__main__':
    subjects = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6])
    dataset = np.copy(subjects)
    cv = LeavePSubjectsOutKFold(subjects, 2)
    for train, test in cv:
        print(subjects[train], subjects[test])

    print(subjectwise_train_test_split(subjects, dataset, n_test_subjects=2))


