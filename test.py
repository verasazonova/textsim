#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:48:40 2014

@author: vera
"""

from gensim.models import TfidfModel, Word2Vec
from gensim import matutils
from corpus.medical import MedicalReviewAbstracts, AugmentedCorpus, prep_arguments
from corpus.reuters import ReutersDataset
from corpus.twits import KenyanTweets
from models.mlda import MldaModel, LdaClassifier, SimDictClassifier
import models.w2v_stacked
from utils import plotutils
from sklearn import cross_validation, svm, metrics
from sklearn.utils import shuffle
import numpy as np
import argparse
from matplotlib import pyplot as plt
import logging
from sklearn import grid_search


def run_classifier(x, y, clf=None, fit_parameters=None):
    n_trials = 1
    n_cv = 2
    print x.shape, y.shape
    logging.info("Testing: fit parameters %s " % (fit_parameters,))
    if clf is None:
        clf = svm.SVC(kernel='linear', C=1)
    scores = np.zeros((n_trials * n_cv))
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))
        x_shuffled, y_shuffled = shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring='accuracy',
                                                                           verbose=2, n_jobs=1,
                                                                           fit_params=fit_parameters)
    return scores

def run_grid_search(x, y, clf=None, parameters=None, fit_parameters=None):
    if clf is None:
        clf = svm.SVC(kernel='linear', C=1)
    if parameters is None:
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    grid_clf = grid_search.GridSearchCV(clf, param_grid=parameters, fit_params=fit_parameters)
    grid_clf.fit(x, y)
    print grid_clf.grid_scores_
    print grid_clf.best_params_
    print grid_clf.best_score_



def run_classifier_test_train_splits(train_data=None, test_data=None, clf=None):
    logging.info("Testing with train and test splits ")
    if clf is None:
        clf = svm.SVC(kernel='linear', C=1)
    x_train, y_train = train_data
    x_test, y_test = test_data
    _ = clf.fit(x_train, y_train)
    y_predicted = clf.predict(x_test)
    return np.array([ metrics.roc_auc_score(y_test, y_predicted) ])


"""
--------------------------------------------------------------------------------
"""

def make_mlda_model(corpus=None, dictionary=None, n_topics=2):
    model = MldaModel(n_topics=n_topics, dictionary=dictionary, bow_corpus=corpus, mallet=True)
    return model

def test_lda(mlda_model=None, tfidf_model=None, corpus=None, dictionary=None, n_topics=2):
    data_tfidf = matutils.corpus2dense(tfidf_model[corpus], num_terms=len(dictionary)).T
    data_mlda = mlda_model.corpus2dense_lda(bow_corpus=corpus, dictionary=dictionary, n_topics=n_topics)
    if data_mlda is None:
        return data_tfidf
    else:
        x_data = np.concatenate((data_tfidf, data_mlda), axis=1)
    return x_data


def test_simdict_classifier(no_below=2, no_above=0.9, simdictname=None):
    return SimDictClassifier(simdictname=simdictname)

"""
--------------------------------------------------------------------------------
"""


def test_parameter(function_name, parameters, target=None, parameter_tosweep=None,
                   value_list=None, filename="test", color='b', logfilename="log.txt",
                   x_data=None, fit_parameters=None, test=None, train=None):
    print "Testing parameter %s in function %s" % (parameter_tosweep, function_name)
    result = []
    with open(logfilename, 'a') as f:
        # f.write("# %s;  %s \n" % (str(parameter_tosweep), str(parameters)))
        for p in value_list:
            parameters[parameter_tosweep] = p
            #x_data = function_name(**parameters)
            print parameters
            clf = function_name(**parameters)
            if test is None or train is None:
                scores = run_classifier(x_data, target, clf=clf, fit_parameters=fit_parameters)
            else:
                scores = run_classifier_test_train_splits(test, train, clf=clf)
            f.write("%s " % (str(p)))
            f.write(" ".join(map(str, scores.tolist())) + "\n")
            f.flush()
            result.append(scores)

    print filename, value_list, result
    plotutils.plot_xy(value_list, result, "n topics", "roc_auc", str(filename) + ".pdf",
                      color=color, s=100)


"""
--------------------------------------------------------------------------------
"""


def get_corpus(filename=None, arguments=None, dataset=None, perword=False, topn=0):
    test = None
    train = None
    if arguments.kt:
        corpus = KenyanTweets(filename)
    elif arguments.categories:
        corpus = ReutersDataset(arguments.categories)
        dataset = "-".join(arguments.categories)
        test = corpus.get_test()
        train = corpus.get_train()
    elif perword:
        filename = dataset + "-" + str(topn) + "pw.txt"
        corpus = AugmentedCorpus(filename)
    else:
        corpus = MedicalReviewAbstracts(filename, ['T', 'A'])
    return corpus

"""
def augments_corpus(arguments=None, topn=2, perword=False, corpus=None):
    if arguments.modelname is not None and topn > 0:
        if perword:
            # corpus already augmented
            x = np.array([text for text in corpus])
            perword_str = "-pw"
        else:
            # augment the corpus per text
            if test is None or train is None:
                augmented_corpus = pmc_w2v.augment_corpus(corpus=corpus, w2v_model=w2v_model, topn=[topn], perword=False)
                x = np.array([[word.lower() for word in text] for text in augmented_corpus])
            else:
                x_train, y_train = train
                x_test, y_test = test

                augmented_train = pmc_w2v.augment_corpus(corpus=x_train, w2v_model=w2v_model, topn=[topn], perword=False)
                x_train = np.array([[word.lower() for word in text] for text in augmented_train])

                augmented_test = pmc_w2v.augment_corpus(corpus=x_test, w2v_model=w2v_model, topn=[topn], perword=False)
                x_test = np.array([[word.lower() for word in text] for text in augmented_test])

                test = (x_test, y_test)
                train = (x_train, y_train)

            perword_str = ""
        dataset += "-" + str(topn) + perword_str

"""

def test_one_file(filename, dataset, topn, perword, w2v_model, arguments):

    corpus = get_corpus(filename=filename, dataset=dataset, arguments=arguments, perword=perword)

    test_type = "none"
    test = None
    train = None

    x = np.array([text for text in corpus])
    y = np.array(corpus.get_target())


    print dataset, perword

    if arguments.test_lda:
        max_n_topics = 20
        test_type = "lda"

        parameters = {"no_below": 2, "no_above": 0.9,
                      "mallet": True, "n_topics": 2}
        parameter_tosweep = "n_topics"
        value_list = range(0, max_n_topics + 1, 4)

        logfilename = dataset + "_" + test_type + ".txt"
        logging.info(logfilename)
        if test is None or train is None:
            test_parameter(LdaClassifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, x_data=x)
        else:
            test_parameter(LdaClassifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, test=test, train=train)


    elif arguments.test_simdict:
        test_type = "simdict"

        if arguments.modelname is not None:
            simdictlist = arguments.modelname
        else:
            print "no model"

        parameters = {"no_below": 2, "no_above": 0.9, "simdictname": None}
        parameter_tosweep = "simdictname"
        value_list = [None] + simdictlist
        logfilename = dataset + "_" + test_type + ".txt"

        test_parameter(test_simdict_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x)

    elif arguments.test_w2v:

        test_type = "word2vec_"
        logfilename = dataset + "_" + test_type + ".txt"

        fit_parameters = {"model": w2v_model}

        parameters = {"learning_rate": 0.06, "n_components": 100, "logistic_C": 6000}
        parameter_tosweep = "n_components"
        value_list = [100]

        test_parameter(models.w2v_stacked.W2VStackedClassifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x,
                       fit_parameters=fit_parameters)


    else:
        test_type = "bow_"

        parameters = {"no_below": 1, "no_above": 1}
        parameter_tosweep = "no_below"
        value_list = [2, 5, 10]

        logfilename = dataset + "_" + test_type + ".txt"
        test_parameter(models.w2v_stacked.BOWClassifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, x_data=x)


    plt.legend()
    plt.title(dataset)
    plt.savefig(dataset + "_tam_" + test_type + ".pdf")

"""
def build_classifier(clf_name):
    if clf_name == "SVM":
        clf = svm.SVC(kernel='linear', C=1)
    elif clf_name == "RBM":

        logistic = linear_model.LogisticRegression()

        rbm = BernoulliRBM(random_state=0, verbose=True)
        rbm.learning_rate = learning_rate
        rbm.n_iter = 20
        # More components tend to give better prediction performance, but larger
        # fitting time
        rbm.n_components = n_components
        logistic.C = logistic_C
        use_svm = use_svm

        if not self.use_svm:
            self.clf = Pipeline(steps=[('rbm', self.rbm), ('logistic', self.logistic)])
        else:

"""

"""
--------------------------------------------------------------------------------
"""


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    parser.add_argument('-d', action='store', nargs="+", dest='dataset', help='Dataset name')
    parser.add_argument('-c', action='store', nargs="+", dest='categories', help='Dataset name')
    parser.add_argument('--topn', action='store', nargs="+", dest='topn', default='0', help='Dataset name')
    parser.add_argument('--model', action='store', nargs="+", dest='modelname', help='Similarity dictionary name')
    parser.add_argument('--lda', action='store_true', dest='test_lda', help='If on test lda features')
    parser.add_argument('--sd', action='store_true', dest='test_simdict', help='knn similarity')
    parser.add_argument('--w2v', action='store_true', dest='test_w2v', help='If on test w2v features')
    parser.add_argument('--w2v-topn', action='store_true', dest='test_w2v_topn', help='If on test w2v features')
    parser.add_argument('--pword', action='store_true', dest='perword', help='whether similar words taken per word')
    parser.add_argument('--kt', action='store_true', dest='kt', help='kenyan twits')
    arguments = parser.parse_args()

    print arguments

    datasets, filenames = prep_arguments(arguments)
    topns = map(int, arguments.topn)
    perword = arguments.perword

    if arguments.modelname is not None and not arguments.test_simdict:
        w2v_model_name = arguments.modelname[0]
        print w2v_model_name

        w2v_model = Word2Vec.load(w2v_model_name)
        w2v_model.init_sims(replace=True)
    else:
        w2v_model = None

    for dataset, filename in zip(datasets, filenames):
        for topn in topns:
            print dataset, filename, topn
            test_one_file(filename, dataset, topn, perword, w2v_model, arguments)


if __name__ == "__main__":
    __main__()