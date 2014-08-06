# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 16:01:11 2014

@author: vera
"""

from gensim import models, corpora, matutils
import numpy as np
from corpus.medical import MedicalReviewAbstracts
from corpus import simdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
import string
import random
import os
import os.path


def make_bow(raw_corpus=None, bow_corpus=None, dictionary=None):
    if bow_corpus is None:
        if raw_corpus is None:
            print "Error no corpus is provided"
            return None
        if dictionary is None:
            dictionary = corpora.Dictionary(raw_corpus)
        bow_corpus = [dictionary.doc2bow(text) for text in raw_corpus]

    if hasattr(bow_corpus, "__len__"):
        n_docs = len(bow_corpus)
    else:
        n_docs = len([1 for _ in bow_corpus])

    return bow_corpus, n_docs


class ModelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, no_below=1, no_above=1, mallet=True, n_topics=2):
        self.clf = svm.SVC(kernel='linear', C=1)
        self.no_below = no_below
        self.no_above = no_above
        self.mallet = mallet
        self.n_topics = n_topics
        self.mallet_path = "/Users/verasazonova/Work/JARS/mallet-2.0.7/bin/mallet"
        self.dictionary = None
        self.tfidf_model = None
        self.model = None

    def fit(self, x, y):
        # self.classes_, indices = np.unique(["foo", "bar", "foo"], return_inverse=True)
        # self.majority_ = np.argmax(np.bincount(indices))
        # self.classes_, y = np.unique(y, return_inverse=True)
        self.build_models(x)
        x_data = self.pre_process(x)
        return self.clf.fit(x_data, y)

    def decision_function(self, x):
        return self.clf.decision_function(self.pre_process(x))

    def predict(self, x):
        # D = self.decision_function(X)
        # return self.classes_[np.argmax(D, axis=1)]

        x_data = self.pre_process(x)
        return self.clf.predict(x_data)
        # return np.repeat(self.classes_[self.majority_], len(X))


class SimDictClassifier(ModelClassifier):
    """
    Builds a join tfidf - similarity matrix model.
    """

    def __init__(self, no_below=1, no_above=1, simdictname=None):
        super(SimDictClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        self.simdictname = simdictname
        self.sim_dict = None

    def build_models(self, x):

        # dictname="/home/vera/Work/TextVisualization/dicts/estrogens-mesh-msr-path.txt"
        self.sim_dict = simdict.SimDictModel(self.simdictname, corpus=x, no_below=self.no_below, no_above=self.no_above)

    ''' Prepare a numpy array of values from the models and tokenized text'''

    def pre_process(self, x):

        bow_corpus = [self.sim_dict.dictionary.doc2bow(text) for text in x]

        if self.sim_dict is None:
            x_data = matutils.corpus2dense(self.sim_dict.tfidf_model[bow_corpus],
                                           num_terms=len(self.sim_dict.dictionary)).T
        else:
            x_data = self.sim_dict.calculate_similarities(self.sim_dict.tfidf_model[bow_corpus])
        return x_data


class LdaClassifier(ModelClassifier):
    """
    Builds a join tfidf / mlda model for a tokenized text.
    X should be an _array_ of token arrays not a generator!!
    """
    # def __init__(self, no_below=1, no_above=1, mallet=True, n_topics=2):
    # super(LdaClassifier,self).__init__(no_below, no_above, mallet, n_topics)

    ''' Build tfidf and mlda models'''

    def build_models(self, x):
        self.dictionary = corpora.Dictionary(x)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)

        bow_corpus = [self.dictionary.doc2bow(text) for text in x]
        self.tfidf_model = models.TfidfModel(bow_corpus, normalize=True)

        if self.n_topics == 0:
            self.model = None
        else:

            if self.mallet:
                self.model = models.LdaMallet(self.mallet_path, corpus=bow_corpus, num_topics=self.n_topics,
                                              id2word=self.dictionary, workers=4,
                                              optimize_interval=10, iterations=1000)

            else:
                self.model = models.LdaModel(bow_corpus, id2word=self.dictionary, num_topics=self.n_topics,
                                             distributed=False,
                                             chunksize=2000, passes=1, update_every=5, alpha='auto',
                                             eta=None, decay=0.5, eval_every=10, iterations=50, gamma_threshold=0.001)

    ''' Prepare a numpy array of values from the models and tokenized text'''

    def pre_process(self, x):

        bow_corpus = [self.dictionary.doc2bow(text) for text in x]
        data_tfidf = matutils.corpus2dense(self.tfidf_model[bow_corpus], num_terms=len(self.dictionary)).T
        if self.model is None:
            return data_tfidf
        else:
            data_lda = matutils.corpus2dense(self.model[bow_corpus], num_terms=len(self.dictionary)).T
            x_data = np.concatenate((data_tfidf, data_lda), axis=1)

        return x_data


class MldaClassifier(ModelClassifier):
    """
    Builds a join tfidf / mlda model for a tokenized text.
    X should be an _array_ of token arrays not a generator!!
    """

    ''' Build tfidf and mlda models'''

    def build_models(self, x):
        self.dictionary = corpora.Dictionary(x)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)

        bow_corpus = [self.dictionary.doc2bow(text) for text in x]
        self.tfidf_model = models.TfidfModel(bow_corpus, normalize=True)
        self.model = MldaModel(n_topics=self.n_topics, dictionary=self.dictionary,
                               bow_corpus=bow_corpus, mallet=self.mallet)

    ''' Prepare a numpy array of values from the models and tokenized text'''

    def pre_process(self, x):

        bow_corpus = [self.dictionary.doc2bow(text) for text in x]
        data_tfidf = matutils.corpus2dense(self.tfidf_model[bow_corpus], num_terms=len(self.dictionary)).T
        data_mlda = self.model.corpus2dense(bow_corpus=bow_corpus, dictionary=self.dictionary,
                                            n_topics=self.n_topics)
        if data_mlda is None:
            return data_tfidf
        else:
            x_data = np.concatenate((data_tfidf, data_mlda), axis=1)

        return x_data


class MldaModel:
    def __init__(self, n_topics=6, raw_corpus=None, dictionary=None, bow_corpus=None, mallet=True):
        self.n_topics = n_topics
        self.models = []
        mallet_path = "/home/vera/mallet-2.0.7/bin/mallet"

        if bow_corpus is None:
            if dictionary is None:
                dictionary = corpora.Dictionary(raw_corpus)
            if raw_corpus is not None:
                bow_corpus = [dictionary.doc2bow(text) for text in raw_corpus]

        if bow_corpus is not None:

            for n in range(2, n_topics + 1):
                print "executing topic %i" % n
                if mallet:
                    tmp_dir = "/home/vera/Work/mallet_tmp/" + ''.join(
                        random.choice(string.ascii_uppercase + string.digits) for _ in range(6)) + "/"
                    while os.path.isdir(tmp_dir):
                        tmp_dir = "/home/vera/Work/mallet_tmp/" + ''.join(
                            random.choice(string.ascii_uppercase + string.digits) for _ in range(6)) + "/"
                    os.makedirs(tmp_dir)
                    self.models.append(models.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=n,
                                                        id2word=dictionary, workers=4, prefix=tmp_dir,
                                                        optimize_interval=10, iterations=1000))

                else:
                    self.models.append(models.LdaModel(bow_corpus, id2word=dictionary, num_topics=n, distributed=False,
                                                       chunksize=2000, passes=1, update_every=5, alpha='auto',
                                                       eta=None, decay=0.5, eval_every=10, iterations=50,
                                                       gamma_threshold=0.001))

    def save(self, filename):
        for i, model in enumerate(self.models):
            model.save("%s.%i" % (filename, i))

    @classmethod
    def load(cls, filename, n_topics):
        new_model = MldaModel(n_topics=n_topics)
        for n in range(2, n_topics + 1):
            new_model.models.append(models.LdaMallet.load("%s.%i" % (filename, n - 2)))
        return new_model

    def corpus2dense_lda(self, raw_corpus=None, bow_corpus=None, dictionary=None, n_topics=2):
        bow_corpus, n_docs = make_bow(raw_corpus, bow_corpus, dictionary)
        if n_topics < 2:
            return None
        data = matutils.corpus2dense(self.models[n_topics - 2][bow_corpus], num_terms=n_topics).T
        return data

    def corpus2dense(self, raw_corpus=None, bow_corpus=None, dictionary=None, n_topics=None):
        bow_corpus, n_docs = make_bow(raw_corpus, bow_corpus, dictionary)

        if n_topics is None:
            n_topics = self.n_topics
        elif n_topics < 2:
            return None

        n_terms = n_topics * (n_topics + 1) / 2 - 1

        data = np.empty((n_docs, n_terms))
        start_ind = 0
        # print "Data ", data.shape
        for n in range(2, n_topics + 1):
            end_ind = start_ind + n
            temp = matutils.corpus2dense(self.models[n - 2][bow_corpus], num_terms=end_ind - start_ind).T
            # print "Temps ", temp.shape
            data[:, start_ind:end_ind] = temp
            # print "From %i, to %i, with n=%i" % (start_ind, end_ind, n)
            start_ind = end_ind

        return data


def __main__():
    filename = "/home/vera/Work/TextVisualization/MedAb_Data/units_all_joined.txt"
    # filename = "/home/vera/Work/TextVisualization/MedAb_Data/units_Estrogens.txt"

    mra = MedicalReviewAbstracts(filename, ['T', 'A', 'M'])
    dictionary = corpora.Dictionary(mra)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in mra]

    n_topics = 20
    model = MldaModel(n_topics=n_topics, dictionary=dictionary, bow_corpus=corpus, mallet=True)

    model.save("./models/all_joined_mlda_20")


if __name__ == "__main__":
    __main__()