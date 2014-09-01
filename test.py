#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 12:48:40 2014

@author: vera
"""

from gensim.models import TfidfModel, Word2Vec
from gensim import corpora, matutils
from corpus.medical import MedicalReviewAbstracts
from models.mlda import MldaModel, MldaClassifier, LdaClassifier, SimDictClassifier
from models.pmc_w2v import W2VModelClassifier
from utils import plotutils
from sklearn import cross_validation, svm
import numpy as np
import argparse
from matplotlib import pyplot as plt
import logging


def run_classifier(x, y, clf=None, fit_parameters=None):
    n_trials = 10
    n_cv = 5
    print x.shape, y.shape
    logging.info("Testing: fit parameters %s " % (fit_parameters,))
    if clf is None:
        clf = svm.SVC(kernel='linear', C=1)
    scores = np.zeros((n_trials * n_cv))
#    scores = np.empty([n_trials * n_cv])
    for n in range(n_trials):
        logging.info("Testing: trial %i or %i" % (n, n_trials))
        skf = cross_validation.StratifiedKFold(y, n_folds=n_cv, random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x, y, cv=skf, scoring='roc_auc',
                                                                           verbose=2, n_jobs=1,
                                                                           fit_params=fit_parameters)
    return scores


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
    return SimDictClassifier(no_below=no_below, no_above=no_above, simdictname=simdictname)


def test_w2v_classifier(no_below=2, no_above=0.9, w2v_model=None, model_type='none', topn=100):
    return W2VModelClassifier(no_below=no_below, no_above=no_above, w2v_model=w2v_model, model_type=model_type,
                              topn=topn)


def test_parameter(function_name, parameters, target=None, parameter_tosweep=None,
                   value_list=None, filename="test", color='b', logfilename="log.txt",
                   x_data=None, fit_parameters=None):
    print "Testing parameter %s in function %s" % (parameter_tosweep, function_name)
    result = []
    with open(logfilename, 'a') as f:
        # f.write("# %s;  %s \n" % (str(parameter_tosweep), str(parameters)))
        for p in value_list:
            parameters[parameter_tosweep] = p
            #x_data = function_name(**parameters)
            print p
            clf = function_name(**parameters)
            scores = run_classifier(x_data, target, clf=clf, fit_parameters=fit_parameters)
            f.write("%s " % (str(p)))
            f.write(" ".join(map(str, scores.tolist())) + "\n")
            f.flush()
            result.append(scores)

    print filename, value_list, result
    plotutils.plot_xy(value_list, result, "n topics", "roc_auc", str(filename) + ".pdf",
                      color=color, s=100)


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    parser.add_argument('-d', action='store', dest='dataset', help='Dataset name')
    parser.add_argument('-m', action='store', dest='model', help='Dataset name')
    parser.add_argument('-s', action='store', nargs="+", dest='simdictname', help='Similarity dictionary name')
    parser.add_argument('--lda', action='store_true', dest='test_lda', help='If on test lda features')
    parser.add_argument('--sd', action='store_true', dest='test_simdict', help='If on test simdict features')
    parser.add_argument('--w2v', action='store_true', dest='test_w2v', help='If on test w2v features')
    parser.add_argument('--w2v-topn', action='store_true', dest='test_w2v_topn', help='If on test w2v features')
    arguments = parser.parse_args()

    print "Hello"

    if (arguments.filename is None) and (arguments.dataset is None):
        dataset = "Estrogens"
        filename = "/Users/verasazonova/no-backup/medab_data/units_Estrogens.txt"
    elif arguments.filename is None:
        dataset = arguments.dataset
        filename = "/Users/verasazonova/no-backup/medab_data/units_" + dataset + ".txt"
    else:
        dataset = "-tasdf-"
        filename = arguments.filename

    print filename

    mra = MedicalReviewAbstracts(filename, ['T', 'A'])
    x = np.array([text for text in mra])
    y = np.array(mra.get_target())
    test_type = "none"

    if arguments.test_lda:
        max_n_topics = 20
        test_type = "lda"

        parameters = {"no_below": 2, "no_above": 0.9,
                      "mallet": True, "n_topics": 2}
        parameter_tosweep = "n_topics"
        value_list = range(0, max_n_topics + 1, 4)

        logfilename = dataset + "_" + test_type + ".txt"
        test_parameter(test_lda_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='g', logfilename=logfilename, x_data=x)

        #logfilename=dataset+"_mldalog.txt"
        #test_parameter(test_mlda_classifier, parameters, target=y,
        #               parameter_tosweep=parameter_tosweep, value_list=value_list,
        #               filename = "mlda", color='r', logfilename=logfilename, x_data=X)

    elif arguments.test_simdict:
        test_type = "simdict"

        if arguments.simdictname is None:
            simdictlist = ["/Users/verasazonova/Work/TextVisualization/dicts/estrogens-mesh-msr-path.txt"]
        else:
            simdictlist = arguments.simdictname

        parameters = {"no_below": 2, "no_above": 0.9, "simdictname": None}
        parameter_tosweep = "simdictname"
        value_list = [None] + simdictlist
        logfilename = dataset + "_" + test_type + ".txt"

        test_parameter(test_simdict_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x)

    elif arguments.test_w2v:
        test_type = "word2vec"

        if arguments.simdictname is None:
            w2v_model_name = "/Users/verasazonova/no-backup/pubmed_central/pmc_100_5"
        else:
            w2v_model_name = arguments.simdictname

        w2v_model = Word2Vec.load(w2v_model_name)
        #w2v_model = None

        parameters = {"no_below": 2, "no_above": 0.9, "w2v_model": None, "model_type": 'none'}
        parameter_tosweep = "model_type"
        value_list = ['none', 'averaged', 'stacked']
        logfilename = dataset + "_" + test_type + ".txt"

        fit_parameters = {"model": w2v_model}
        test_parameter(test_w2v_classifier, parameters, target=y,
                       parameter_tosweep=parameter_tosweep, value_list=value_list,
                       filename=test_type, color='b', logfilename=logfilename, x_data=x,
                       fit_parameters=fit_parameters)

    elif arguments.test_w2v_topn:
        test_type = "word2vec_topn"

        if arguments.simdictname is None:
            w2v_model_name = "/Users/verasazonova/no-backup/pubmed_central/pmc_100_5"
        else:
            w2v_model_name = arguments.simdictname

        w2v_model = Word2Vec.load(w2v_model_name)
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

    plt.legend()
    plt.title(dataset)
    plt.savefig(dataset + "_tam_" + test_type + ".pdf")

    '''
    parameters = {"raw_corpus": mra, "no_above":1, "no_below":1}
    parameter_tosweep = "no_below"
    value_list = range(1,5)
    test_parameter(test_tfidf, parameters, target=mra.get_target(),
                   parameter_tosweep=parameter_tosweep, value_list=value_list,
                   filename = "tfidf", color='b')

    parameter_tosweep = "no_above"
    value_list = [1, 0.9, 0.5, 0.3]
    test_parameter(test_tfidf, parameters, target=mra.get_target(),
                   parameter_tosweep=parameter_tosweep, value_list=value_list,
                   filename = "tfidf", color='k')


    max_n_topics=20

    tfidf_model, corpus, dictionary = make_tfidf_model(raw_corpus=mra, no_below=2, no_above=0.9)

    if arguments.model is None:
        mlda_model = make_mlda_model(corpus=corpus, dictionary=dictionary, n_topics=max_n_topics)
        mlda_model.save("./models/" + dataset+"_mlda_"+str(max_n_topics))
    else:
        mlda_model = MldaModel.load(arguments.model, max_n_topics)


    parameters = {"mlda_model": mlda_model, "tfidf_model":tfidf_model,
                  "corpus": corpus, "dictionary":dictionary, "n_topics":2}
    parameter_tosweep = "n_topics"
    value_list = range(0, max_n_topics+1, 2)

    logfilename=dataset+"_ldalog.txt"
    test_parameter(test_lda, parameters, target=mra.get_target(),
                   parameter_tosweep=parameter_tosweep, value_list=value_list,
                   filename = "lda", color='g', logfilename=logfilename)


    logfilename=dataset+"_mldalog.txt"
    test_parameter(test_mlda, parameters, target=mra.get_target(),
                   parameter_tosweep=parameter_tosweep, value_list=value_list,
                   filename = "mlda", color='r',logfilename=logfilename)


    plt.pyplot.legend()
    plt.pyplot.title(dataset)
    plt.pyplot.savefig(dataset+"_tam_mlda.pdf")

#    plt.pyplot.savefig("test.pdf")

#    model = LdaModel(corpus, id2word=dictionary, num_topics=10, distributed=False,
#                     chunksize=2000, passes=10, update_every=5, alpha='auto',
#                     eta=None, decay=0.5, eval_every=10, iterations=50, gamma_threshold=0.001)
#
#    print matutils.corpus2dense(model[corpus], num_terms=10)
'''


if __name__ == "__main__":
    __main__()