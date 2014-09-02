__author__ = 'verasazonova'

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.datasets import make_classification

def __main__():

    x, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1)
    n_trials = 10
    n_cv = 5
    clf = svm.SVC(kernel='linear', C=1)
    scores = np.zeros((n_trials * n_cv))
#    scores = np.empty([n_trials * n_cv])
    for n in range(n_trials):
        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring='accuracy',
                                                                           verbose=0, n_jobs=n_cv)
    print scores

if __name__ == "__main__":
    __main__()