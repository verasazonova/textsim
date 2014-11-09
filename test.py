#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:48:40 2014

@author: vera
"""

from gensim.models import TfidfModel, Word2Vec
from gensim import corpora, matutils
from corpus.medical import MedicalReviewAbstracts, AugmentedCorpus
from corpus.reuters import ReutersDataset
from corpus.twits import KenyanTweets
from corpus import simdict
from models.mlda import MldaModel, MldaClassifier, LdaClassifier, SimDictClassifier
import models.w2v_stacked
from models import pmc_w2v
from utils import plotutils
from sklearn import cross_validation, svm, metrics, neighbors
from sklearn.utils import shuffle
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt
import logging
from sklearn import grid_search
import math


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

def make_tfidf_model(raw_corpus=None, no_below=1, no_above=1):
    dictionary = corpora.Dictionary(raw_corpus)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(text) for text in raw_corpus]
    tfidf_model = TfidfModel(corpus, normalize=True)
    return tfidf_model, corpus, dictionary


def make_mlda_model(corpus=None, dictionary=None, n_topics=2):
    model = MldaModel(n_topics=n_topics, dictionary=dictionary, bow_corpus=corpus, mallet=True)
    return model


def test_tfidf(raw_corpus=None, no_above=1, no_below=1):
    tfidf_model, corpus, dictionary = make_tfidf_model(raw_corpus=raw_corpus, no_below=no_below, no_above=no_above)
    data_tfidf = matutils.corpus2dense(tfidf_model[corpus], num_terms=len(dictionary)).T
    return data_tfidf


def test_lda(mlda_model=None, tfidf_model=None, corpus=None, dictionary=None, n_topics=2):
    data_tfidf = matutils.corpus2dense(tfidf_model[corpus], num_terms=len(dictionary)).T
    data_mlda = mlda_model.corpus2dense_lda(bow_corpus=corpus, dictionary=dictionary, n_topics=n_topics)
    if data_mlda is None:
        return data_tfidf
    else:
        x_data = np.concatenate((data_tfidf, data_mlda), axis=1)
    return x_data


def test_mlda(mlda_model=None, tfidf_model=None, corpus=None, dictionary=None, n_topics=2):
    data_tfidf = matutils.corpus2dense(tfidf_model[corpus], num_terms=len(dictionary)).T
    data_mlda = mlda_model.corpus2dense(bow_corpus=corpus, dictionary=dictionary, n_topics=n_topics)
    if data_mlda is None:
        return data_tfidf
    else:
        x_data = np.concatenate((data_tfidf, data_mlda), axis=1)
    return x_data


def test_lda_classifier(no_below=2, no_above=0.9, mallet=True, n_topics=3):
    return LdaClassifier(no_below=no_below, no_above=no_above, mallet=mallet, n_topics=n_topics)


def test_mlda_classifier(no_below=2, no_above=0.9, mallet=True, n_topics=3):
    return MldaClassifier(no_below=no_below, no_above=no_above, mallet=mallet, n_topics=n_topics)


def test_simdict_classifier(no_below=2, no_above=0.9, simdictname=None):
    return SimDictClassifier(simdictname=simdictname)


def test_w2v_classifier(no_below=2, no_above=0.9):
    return models.w2v_stacked.W2VStackedClassifier(no_below=no_below, no_above=no_above)


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
            print p
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


def prep_arguments(arguments):

    prefix = os.environ.get("MEDAB_DATA")
    datasets = []
    filenames = []
    if (arguments.filename is None) and (arguments.dataset is None):
        datasets = ["Estrogens"]
        filenames = [prefix + "/units_Estrogens.txt"]
    elif arguments.filename is None:
        datasets = arguments.dataset
        filenames =  [prefix + "/units_" + dataset + ".txt" for dataset in datasets]
    else:
        exit()


    topn = map(int, arguments.topn)
    perword = arguments.perword
    return datasets, filenames, topn, perword


def test_one_file(filename, dataset, topn, perword, w2v_model, arguments):

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
    x = None

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

    elif test is None or train is None:
        x = np.array([text for text in corpus])

    test_type = "none"
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
            test_parameter(test_lda_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, x_data=x)
        else:
            test_parameter(test_lda_classifier, parameters, target=y,
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
        #parameters = {"no_below": [2], "no_above": [0.5], "n_components": [10, 30,100],
        #              "logistic_C": [1, 100, 1000], "learning_rate": [0.06]}

        logfilename = dataset + "_" + test_type + ".txt"

        fit_parameters = {"model": w2v_model}

        parameters = {"no_below": 2, "no_above": 0.9, "use_svm": True}
        parameter_tosweep = "no_below"
        value_list = [2]

        #clf = models.w2v_stacked.W2VStackedClassifier()
        #run_grid_search(x, y, clf=clf, parameters=parameters, fit_parameters=fit_parameters)

        test_parameter(models.w2v_stacked.W2VStackedClassifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x,
                       fit_parameters=fit_parameters)


    elif arguments.test_w2v_topn:
        test_type = "word2vec_topn"

        if arguments.modelname is not None:
            w2v_model_name = arguments.modelname[0]
        else:
            exit()

        w2v_model = Word2Vec.load(w2v_model_name)
        w2v_model.init_sims(replace=True)
        #w2v_model = None

        parameters = {"no_below": 2, "no_above": 0.9, "w2v_model": None, "model_type": 'augmented', "topn": 200}
        parameter_tosweep = "topn"
        value_list = [200, 300]
        logfilename = dataset + "_" + test_type + ".txt"

        fit_parameters = {"model": w2v_model}
        test_parameter(test_w2v_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x,
                       fit_parameters=fit_parameters)

    else:
        max_n_topics = 0
        test_type = "none"

        parameters = {"no_below": 1, "no_above": 1,
                      "mallet": True, "n_topics": 0}
        parameter_tosweep = "n_topics"
        value_list = range(0, max_n_topics + 1, 4)

        logfilename = dataset + "_" + test_type + ".txt"
        test_parameter(test_lda_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, x_data=x)


    plt.legend()
    plt.title(dataset)
    plt.savefig(dataset + "_tam_" + test_type + ".pdf")


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

    datasets, filenames, topns, perword = prep_arguments(arguments)

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