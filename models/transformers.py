__author__ = 'verasazonova'

import logging
from corpus.medical import word_valid
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim import corpora, models, matutils
from gensim.models.doc2vec import LabeledSentence
import re


class BOWModel(BaseEstimator, TransformerMixin):
    def __init__(self, no_below=2, no_above=0.9):
        self.no_below = no_below
        self.no_above = no_above
        logging.info("BOW classifier: initialized with no_below %s and no_above %s " % (self.no_below, self.no_above))


    def fit(self, X, y=None):
        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.bow = [self.dictionary.doc2bow(text) for text in X]
        self.tfidf = models.TfidfModel(self.bow, dictionary=self.dictionary, normalize=True)
        return self


    def transform(self, X):
        x_tfidf = self.tfidf[[self.dictionary.doc2bow(text) for text in X]]
        x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        logging.info("Returning data of shape %s " % (x_data.shape,))
        return x_data


def sents_to_labeled(X):
    return [LabeledSentence(text, [str(cnt)]) for cnt, text in enumerate(X)]


class D2VModel(BaseEstimator, TransformerMixin):
    def __init__(self, d2v_model=None):
        self.d2v_model = d2v_model
        logging.info("D2V")


    def fit(self, X, y=None):
        logging.info("D2V: building a doc2vector model")
        if self.d2v_model is None:
            self.d2v_model = models.Doc2Vec(sentences=sents_to_labeled(X), size=50, alpha=0.025, window=8, min_count=3, sample=0, seed=1,
                                        workers=4, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0,
                                        train_words=True, train_lbls=True)
        logging.info("Model built %s " % (self.d2v_model, ))
        return self


    def transform(self, X):
        logging.info("D2V: pre-processing data of shape %s " % (X.shape, ))
        n_docs = len(X)
        if self.d2v_model is not None:
            self.d2v_model.train_words = False
            self.d2v_model.train(sents_to_labeled(X))

            d2v_length = len(self.d2v_model['test'])
            logging.info("D2V:  w2v-len %s " % (d2v_length, ))
            data = np.zeros((n_docs, d2v_length))

            for doc_cnt, text in enumerate(X):
                data[doc_cnt, :] = self.d2v_model[str(doc_cnt)]

            logging.info("W2V Stacked: returning pre-processed data of shape %s" % (data.shape, ))
        else:
            data = np.zeros((n_docs, 1))
        print data
        return data


class W2VStackedModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None):
        self.w2v_model = w2v_model
        logging.info("W2v stacked classifier")


    def fit(self, X, y=None):
        logging.info("W2V: building a model")
        self.max_textlen = max( [len(text) for text in X])
        logging.info("Max text length is %s " % (self.max_textlen, ))
        return self


    def transform(self, X):
        logging.info("W2V Stacked: pre-processing data of shape %s " % (X.shape, ))
        n_docs = len(X)
        if self.w2v_model is not None:
            w2v_length = len(self.w2v_model['test'])
            logging.info("W2V Stacked: max_textlen %s, w2v-len %s " % (self.max_textlen, w2v_length))
            data = np.zeros((n_docs, self.max_textlen * w2v_length))
            for doc_cnt, text in enumerate(X):
                cnt = 0
                for word in text:
                    if word in self.w2v_model:
                        data[doc_cnt, cnt:cnt + w2v_length] = self.w2v_model[word]
                    cnt += w2v_length
                    if cnt >= self.max_textlen * w2v_length:
                        break
            logging.info("W2V Stacked: returning pre-processed data of shape %s" % (data.shape, ))
        else:
            data = np.zeros((n_docs, 1))
        return data


class W2VStackedBOWModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None, no_above=0.9, no_below=3):
        self.w2v_model = w2v_model
        self.no_above = no_above
        self.no_below = no_below
        logging.info("W2v stacked bow classifier")


    def fit(self, X, y=None):
        logging.info("W2V stacked bow: building a model")
        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.bow = [self.dictionary.doc2bow(text) for text in X]
        self.tfidf = models.TfidfModel(self.bow, dictionary=self.dictionary, normalize=True)
        return self


    def transform(self, X):
        logging.info("W2V Stacked BOW: pre-processing data of shape %s " % (X.shape, ))
        n_docs = len(X)
        n_words = len(self.dictionary)
        bow = [self.dictionary.doc2bow(text) for text in X]
        if self.w2v_model is not None:
            w2v_length = len(self.w2v_model['test'])
            logging.info("W2V Stacked BOW: w2v-len %s " % (w2v_length,))
            data = np.zeros((n_docs, n_words * w2v_length))
            for doc_cnt, text in enumerate(bow):
                cnt = 0
                for word_id, weight in text:
                    word = self.dictionary[word_id]
                    if word in self.w2v_model:
                        data[doc_cnt, cnt:cnt + w2v_length] = weight * self.w2v_model[word]
                    cnt += w2v_length
            logging.info("W2V Stacked: returning pre-processed data of shape %s" % (data.shape, ))
        else:
            data = np.zeros((n_docs, 1))
        return data



class W2VAveragedModel(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None):
        self.w2v_model = w2v_model
        logging.info("W2v averaged classifier %s " % self.w2v_model )


    def fit(self, X, y=None):
        logging.info("W2V: got a model %s " % (self.w2v_model,))
        return self


    def transform(self, X):
        n_docs = len(X)
        if self.w2v_model is not None:
            logging.info("W2V Averaged: pre-processing data of shape %s " % (X.shape, ))
            w2v_length = len(self.w2v_model['test'])
            logging.info("W2V Averaged: w2v-len %s " % (w2v_length, ))
            data = np.zeros((n_docs, w2v_length))
            for doc_cnt, text in enumerate(X):
                for word in text:
                    if word in self.w2v_model:
                        data[doc_cnt] += self.w2v_model[word]
            logging.info("W2V Averaged: returning pre-processed data of shape %s" % (data.shape, ))
        else:
            logging.info("W2V Averaged: no model was provided." )
            data = np.zeros((n_docs, 1))
        return data


class W2VAugmentModel(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None, w2v_model=None):
        self.topn = topn
        self.w2v_model = w2v_model
        logging.info("W2v stacked classifier")


    def fit(self, X, y=None):
        logging.info("W2V: building a model")
        logging.info("W2V: model assigned %s with topn %s" % (self.w2v_model, self.topn))
        return self


    def transform(self, X):
        logging.info("W2V Augmented: augmenting the text %s" % (X.shape, ))
        if self.w2v_model is None:
            augmented_corpus = X[:]
        else:
            augmented_corpus = []
            for text in X:
                words_in_model = [word for word in text if word in self.w2v_model]
                augmented_text = [word for word in text]
                sim_words = [re.sub(r"\W", "_", tup[0]) for
                             tup in self.w2v_model.most_similar(positive=words_in_model, topn=self.topn)
                             if word_valid(tup[0])]
                augmented_text += sim_words
                augmented_corpus.append(augmented_text)
        a = np.array(augmented_corpus)
        print a.shape
        return a


class LDAModel(BaseEstimator, TransformerMixin):

    def __init__(self, topn=None, no_below=1, no_above=1, mallet=True):
        self.topn = topn
        self.no_above = no_above
        self.no_below = no_below
        self.mallet = mallet
        self.mallet_path = "/Users/verasazonova/no-backup/JARS/mallet-2.0.7/bin/mallet"


    def fit(self, X, y=None):
        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]

        if self.mallet:
            self.model = models.LdaMallet(self.mallet_path, corpus=bow_corpus, num_topics=self.topn,
                                          id2word=self.dictionary, workers=4,
                                          optimize_interval=10, iterations=1000)
        else:
            self.model = models.LdaModel(bow_corpus, id2word=self.dictionary, num_topics=self.topn,
                                         distributed=False,
                                         chunksize=2000, passes=1, update_every=5, alpha='auto',
                                         eta=None, decay=0.5, eval_every=10, iterations=50, gamma_threshold=0.001)
        return self


    def transform(self, X):

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]
        x_data = matutils.corpus2dense(self.model[bow_corpus], num_terms=len(self.dictionary)).T
        return x_data
