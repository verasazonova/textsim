#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:48:40 2014

@author: vera
"""

from gensim.models import Word2Vec, Doc2Vec
from corpus.medical import get_filename
from models.transformers import W2VStackedModel, BOWModel, W2VAveragedModel, LDAModel, W2VAugmentModel, \
    D2VModel, W2VStackedBOWModel
from corpus.medical import MedicalReviewAbstracts
import numpy as np
import argparse
import logging
from sklearn.utils import shuffle
from sklearn import grid_search, linear_model, svm, cross_validation, neighbors
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neural_network import BernoulliRBM
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def run_grid_search(x, y, clf=None, parameters=None, fit_parameters=None):
    if clf is None:
        raise Exception("No classifier passed")
    if parameters is None:
        raise Exception("No parameters passed")
    print parameters
    grid_clf = grid_search.GridSearchCV(clf, param_grid=parameters, fit_params=fit_parameters,
                                        iid=False, cv=10, refit=True)
    grid_clf.fit(x, y)
    print grid_clf.grid_scores_
    print grid_clf.best_params_
    print grid_clf.best_score_

    return grid_clf.best_score_

def run_cv_classifier(x, y, clf=None):
    n_trials = 50
    n_cv = 2
    scores = np.zeros((n_trials * n_cv))
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))

        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring='accuracy',
                                                                           verbose=2, n_jobs=1)
    print scores, scores.mean(), scores.std()
    return scores.mean()


def run_tsne(x, y, clf=None):

    data = clf.fit_transform(x)
    plt.scatter(data[:, 0], data[:, 1], c=y, s=20)
    plt.savefig("output.pdf")
    return None


def test_one_file(dataset, model, model_type, clf, parameters_clf, d2v_corpus=None, run_type='cls'):

    filename = get_filename(dataset)
    corpus = MedicalReviewAbstracts(filename, ['T', 'A'], labeled=True)

    x = np.array([text for text in corpus])
    y = np.array(corpus.get_target())

    parameters = {}
    fit_parameters = {}

    if model_type == "w2v_stacked":

        clf_pipeline = Pipeline([
            ('w2v', W2VStackedBOWModel(w2v_model=None)),
            ('clf', clf) ])

        parameters = {'w2v__w2v_model': (model, None )}


    elif model_type == "w2v_avg":

        clf_pipeline = Pipeline([
            ('w2v_avg', W2VAveragedModel(w2v_model=None)),
            ('clf', clf) ])


    elif model_type == "w2v_augment":

        clf_pipeline = Pipeline([
            ('w2v_augm', W2VAugmentModel(topn=2)),
            ('bow', BOWModel(no_above=0.9, no_below=2)),
            ('clf', clf) ])

        parameters = {
                      'w2v_augm__topn': (10, 30, 100, 300),
                      'w2v_augm__w2v_model': (model,),
                      'bow__no_below': (2, 3, 5, 10),
                      'bow__no_above': (0.2, 0.3, 0.5, 0.9),}

    elif model_type == "lda":

        clf_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('lda', LDAModel(no_below=3, no_above=0.5, mallet=False)),
                ('bow', BOWModel(no_below=3, no_above=0.5))
                ])),
            ('clf', clf) ])

        parameters = {
                      'features__lda__topn': (4, 8, 12, 16, 20, 25),
                      }

    elif model_type == "bow":

        clf_pipeline = Pipeline([
            ('bow', BOWModel(no_above=0.9, no_below=2)),
            ('clf', clf) ])

        parameters = {}
        # parameters = {'bow__no_below': (2, 3, 5, 10),
         #              'bow__no_above': (0.2, 0.3, 0.5, 0.9)}
        fit_parameters = {}

    elif model_type == "mixed":

        clf_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('w2v', W2VStackedBOWModel(w2v_model=None)),
                ('bow', BOWModel(no_below=3, no_above=0.5))
                ])),
            ('clf', clf) ])

#        fit_parameters = {'features__w2v__model': w2v_model}
        parameters = {'features__w2v__w2v_model': (model, None )}

    elif model_type == "d2v":

        clf_pipeline = Pipeline([
            ('features', FeatureUnion([
                ('d2v', D2VModel(d2v_model=None, corpus=d2v_corpus, alpha=0.025, size=100, window=5, min_count=8,
                             min_alpha=0.0001, num_iters=1, initial_w2v=model)),
                ('bow', BOWModel(no_below=2, no_above=0.9))
                ])),
            ('clf', clf) ])

        parameters = {}

#        parameters = {'features__d2v__initial_w2v': [None, model]}
#                      'features__d2v__num_iters': [1, 10],
#                      'features__d2v__min_count': [3, 5, 8, 10]
#                        }

    else:

        raise Exception("Unknown test type")


    parameters.update(parameters_clf)

    if run_type == 'tsne':
        mean = run_tsne(x, y, clf=clf_pipeline)
    elif parameters:
        mean = run_grid_search(x, y, clf=clf_pipeline, parameters=parameters, fit_parameters=fit_parameters)
    else:
        mean = run_cv_classifier(x, y, clf=clf_pipeline)

    return mean



def build_classifier(clf_name):

    clf = None
    parameters = {}

    if clf_name == "svm":
        clf = svm.SVC(kernel='linear', C=10)
        parameters = {}

    elif clf_name == "knn":
        clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='brute', leaf_size=30,
                                             metric='cosine', metric_params=None)

    elif clf_name == "rmb":
        logistic = linear_model.LogisticRegression()
        rbm = BernoulliRBM(random_state=0, verbose=True)
        rbm.learning_rate = 0.01
        rbm.n_iter = 20
        rbm.n_components = 100
        logistic.C = 6000
        clf = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
        #parameters = {'clf__C': (1, 10)}

    elif clf_name == "tsne":
        clf = TSNE(n_components=2, init='random', metric='cosine')

    return clf, parameters

"""
--------------------------------------------------------------------------------
"""


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    parser.add_argument('-d', action='store', nargs="+", dest='datasets', help='Dataset name')
    parser.add_argument('--topn', action='store', nargs="+", dest='topns', default='0', help='Dataset name')
    parser.add_argument('--corpus', action='store', nargs="+", dest='corpusname', help='Similarity dictionary name')
    parser.add_argument('--model', action='store', nargs="+", dest='modelname', help='Similarity dictionary name')
    parser.add_argument('--type', action='store', dest='model_type', help='Data filename')
    parser.add_argument('--cls', action='store', dest='cls_type', help='Data filename')
    arguments = parser.parse_args()

    print arguments

    topns = map(int, arguments.topns)
    model_type = arguments.model_type
    corpus = None
    w2v_model = None

    if arguments.modelname is not None:
        if arguments.model_type == "d2v":
            corpus_name = arguments.corpusname[0]
            w2v_model = arguments.modelname[0]
            corpus = MedicalReviewAbstracts(corpus_name, ['T', 'A'], labeled=True)
        else:
            w2v_model_name = arguments.modelname[0]
            w2v_model = Word2Vec.load(w2v_model_name)
            w2v_model.init_sims(replace=True)
    else:
        w2v_model = None

    clf, parameters = build_classifier(arguments.cls_type)

    for dataset in arguments.datasets:
        for topn in topns:
            print dataset, topn, model_type
            #baseline = test_one_file(dataset, w2v_model, 'bow', clf, parameters, d2v_corpus=corpus)
            new = test_one_file(dataset, w2v_model, model_type, clf, parameters, d2v_corpus=corpus,
                                run_type=arguments.cls_type)
            #print "%s: %s; bow %s" % (model_type, new, baseline)

if __name__ == "__main__":
    __main__()