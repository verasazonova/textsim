__author__ = 'verasazonova'

import logging
from corpus.medical import word_valid
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import check_random_state, shuffle
from gensim import corpora, models, matutils
from gensim.models.doc2vec import LabeledSentence
import re
from sklearn import preprocessing

'''
sklearn-compatible BOW-gensim model
'''
class BOWModel(BaseEstimator, TransformerMixin):
    def __init__(self, no_below=2, no_above=0.9):
        self.no_below = no_below
        self.no_above = no_above
        logging.info("BOW classifier: initialized with no_below %s and no_above %s " % (self.no_below, self.no_above))


    def fit(self, X, y=None):

        X = [text.words for text in X]

        self.dictionary = corpora.Dictionary(X)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.tfidf = models.TfidfModel([self.dictionary.doc2bow(text) for text in X], id2word=self.dictionary, normalize=True)
        return self


    def transform(self, X):
        X = [text.words for text in X]
        x_tfidf = self.tfidf[[self.dictionary.doc2bow(text) for text in X]]
        x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        logging.info("Returning data of shape %s " % (x_data.shape,))
        print x_data
        return x_data


def sents_to_labeled(X):
    return [LabeledSentence(text, [str(cnt)]) for cnt, text in enumerate(X)]



def train_model(model, num_iters, X, train_type):
    alpha = model.alpha
    min_alpha = model.min_alpha
    if int(num_iters) == 1:
        model.train(X)
    else:
        X1 = [x for x in X]
        #inc = 0.002
        inc = (model.alpha - model.min_alpha) / (num_iters)
        print len(X1), inc, model, num_iters, train_type
        for i in range(0, int(num_iters)+1):
            print model.alpha
            if train_type == "fixed":
                model.min_alpha = model.alpha
                model.train(X1) #shuffle(X1, random_state=i))
                model.alpha -=  inc
            else:
                model.alpha = alpha
                model.train(shuffle(X1, random_state=i))
        model.alpha = alpha
        model.min_alpha = min_alpha


'''
sklearn-compatible D2V-gensim model
'''
class D2VModel(BaseEstimator, TransformerMixin):
    def __init__(self, d2v_model=None, corpus=None, alpha=0.05, size=100, window=5, initial_w2v=None, min_count=5,
                 min_alpha=0.0001, num_iters=1, sample=0, negative=0, two_models=True, dm=1, hs=1, seed=0,
                 num_iters_words=1, train_type="fixed", alpha_words=None):
        self.d2v_model = d2v_model
        self.corpus = corpus
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.size = size
        self.window = window
        self.min_count = min_count
        self.initial_w2v = initial_w2v
        self.num_iters = num_iters
        self.d2v_model2 = None
        self.sample = sample
        self.negative = negative
        self.hs = hs
        self.dm = dm
        self.seed = seed
        self.train_type = train_type
        self.num_iters_words = num_iters_words
        self.alpha_words = alpha_words
        self.two_models = two_models
        logging.info("D2V")


    def fit(self, X, y=None):
        logging.info("D2V: got a doc2vector model %s ", (self.d2v_model, ))
        if self.num_iters_words is None:
            self.num_iters_words = self.num_iters
        else:
            self.num_iters_words = self.num_iters_words
        if self.alpha_words is None:
            self.alpha_words = self.alpha
        else:
            self.alpha_words = self.alpha_words

        if self.d2v_model is None and self.corpus is not None:
            logging.info("D2V: building a model with size %s, window %s, alpha %s on corpus %s" %
                         (self.size, self.window, self.alpha, self.corpus))
            self.d2v_model = models.Doc2Vec(sentences=None, size=self.size, alpha=self.alpha, window=self.window,
                                            min_count=self.min_count, sample=self.sample, seed=self.seed,
                                            workers=4, min_alpha=self.min_alpha, dm=self.dm, hs=self.hs,
                                            negative=self.negative,
                                            dm_mean=0,
                                            train_words=True, train_lbls=False, initial=self.initial_w2v)

            self.d2v_model.build_vocab(self.corpus)

            '''
            Train words
            '''
            self.d2v_model.train_words = True
            self.d2v_model.train_lbls = False
            self.d2v_model.alpha = self.alpha_words
            train_model(self.d2v_model, self.num_iters_words, self.corpus, self.train_type)

            if self.two_models:
                self.d2v_model2 = models.Doc2Vec(sentences=None, size=self.size, alpha=self.alpha, window=self.window,
                                                min_count=self.min_count, sample=self.sample, seed=self.seed,
                                                workers=4, min_alpha=self.min_alpha, dm=abs(1-self.dm), hs=self.hs,
                                                negative=self.negative,
                                                dm_mean=0,
                                                train_words=True, train_lbls=False, initial=self.initial_w2v)

                self.d2v_model2.build_vocab(self.corpus)

                self.d2v_model2.train_words = True
                self.d2v_model2.train_lbls = False
                self.d2v_model2.alpha = self.alpha_words
                train_model(self.d2v_model2, self.num_iters_words, self.corpus, self.train_type)

            logging.info("Model 2 built %s " % (self.d2v_model2, ))
        print "Fitted"
        return self


    def transform(self, X):
        logging.info("D2V: pre-processing data of shape %s " % (X.shape, ))
        n_docs = len(X)
        if self.d2v_model is not None:

            '''
            Train sentences
            '''
            self.d2v_model.train_words = False
            self.d2v_model.train_lbls = True
            self.d2v_model.alpha = self.alpha
            train_model(self.d2v_model, self.num_iters, X, self.train_type)

            if self.d2v_model2 is not None:
                self.d2v_model2.train_words = False
                self.d2v_model2.train_lbls = True
                self.d2v_model2.alpha = self.alpha
                train_model(self.d2v_model2, self.num_iters, X, self.train_type)

            d2v_length = self.d2v_model.layer1_size
            logging.info("D2V:  w2v-len %s " % (d2v_length, ))
            if self.d2v_model2 is None:
                data = np.zeros((n_docs, d2v_length))
            else:
                data = np.zeros((n_docs, 2*d2v_length))

            for doc_cnt, text in enumerate(X):
                if text.labels[0] in self.d2v_model:
                    data[doc_cnt, 0:d2v_length] = self.d2v_model[text.labels[0]]
                    if self.d2v_model2 is not None:
                        data[doc_cnt, d2v_length:] = self.d2v_model2[text.labels[0]]

            logging.info("D2V: returning pre-processed data of shape %s" % (data.shape, ))
        else:
            data = np.zeros((n_docs, 1))
        print "Transformed"
        return preprocessing.normalize(data)


class ReadVectorsModel(BaseEstimator, TransformerMixin):

    def __init__(self, filename):
        self.vectors = {}
        self.filename = filename

    def fit(self, X, y=None):
        with open(self.filename, 'r') as f:
            for line in f:
                data = line.split()
                label = data[0]
                self.vectors[str(label)] = np.array(map(float, data[1:]))
        return self

    def transform(self, X):
        data = [self.vectors[text.labels[0]] for text in X]
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


'''
sklearn-compatible LDA-gensim model
'''
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

        for topic in  self.model.show_topics(num_topics=self.topn, num_words=20, formatted=False):
            for word in topic:
                print word[1] + " ",
            print ""
        return self


    def transform(self, X):

        bow_corpus = [self.dictionary.doc2bow(text) for text in X]
        x_data = matutils.corpus2dense(self.model[bow_corpus], num_terms=self.topn).T
        return x_data



class W2VList(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_model=None):
        self.w2v_model = w2v_model


    def fit(self, X):
        return self

    def transform(self, X):
        n_docs = len(X)
        data = []
        if self.w2v_model is None:
            raise Exception("No word2vec model provided")

        for doc_cnt, text in enumerate(X):
            cnt = 0
            vector_list = []
            for word in text:
                if word in self.w2v_model:
                    vector_list.append( self.w2v_model[word] )
                else:
                    #vector_list.append( np.ones(self.w2v_model.layer1_size, ))
                    vector_list.append( self.w2v_model['test'])
            data.append(vector_list)
        return data



class CRPClusterer(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X):
        self.random_state_ = check_random_state(self.random_state)
        return self


    # X: an array of real vectors
    def transform(self, X):
        clusters = []
        for vecs in X:
            #clusterVec = [np.zeros(vecs[0].shape,)]         # tracks sum of vectors in a cluster
            #clusterIdx = [[0]]         # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
            clusterVec = []
            clusterIdx = []
            ncluster = 0
            # probablity to create a new table if new customer
            # is not strongly "similar" to any existing table
            pnew = 1.0/ (1 + ncluster)
            N = len(vecs)
            rands = self.random_state_.rand(N)
            #rands = rands / np.max(rands)

            for i, v in enumerate(vecs):
                maxSim = -np.inf
                maxIdx = 0
                #v = vecs[i]
                for j in range(ncluster):
                    sim = cosine_similarity(v, clusterVec[j])[0][0]
                    if sim >= maxSim:
                        maxIdx = j
                        maxSim = sim
                if maxSim < pnew:
                    if rands[i] < pnew:
                        clusterVec.append(v)
                        clusterIdx.append([i])
                        ncluster += 1
                        pnew = 1.0 / (1 + ncluster)
                    continue
                clusterVec[maxIdx] = clusterVec[maxIdx] + v
                clusterIdx[maxIdx].append(i)
            clusters.append(clusterIdx)
        return clusters



class CRPW2VClusterer(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None, w2v_model=None):
        self.random_state = random_state
        self.model = w2v_model

    def fit(self, X):
        self.random_state_ = check_random_state(self.random_state)
        return self


    # X: an array of real vectors
    def transform(self, X):
        clusters = []
        for text in X:
            #clusterVec = [np.zeros(vecs[0].shape,)]         # tracks sum of vectors in a cluster
            #clusterIdx = [[0]]         # array of index arrays. e.g. [[1, 3, 5], [2, 4, 6]]
            clusterVec = []
            clusterIdx = []
            ncluster = 0
            # probablity to create a new table if new customer
            # is not strongly "similar" to any existing table
            pnew = 1.0/ (1 + ncluster)
            N = len(text)
            rands = self.random_state_.rand(N)
            #rands = rands / np.max(rands)

            for i, word in enumerate(text):
                maxSim = -np.inf
                maxIdx = 0
                #v = vecs[i]
                for j in range(ncluster):
                    sim = self.model.similarity(word, clusterVec[j])
                    if sim >= maxSim:
                        maxIdx = j
                        maxSim = sim
                if maxSim < pnew:
                    if rands[i] < pnew:
                        clusterVec.append(v)
                        clusterIdx.append([i])
                        ncluster += 1
                        pnew = 1.0 / (1 + ncluster)
                    continue
                clusterVec[maxIdx] = clusterVec[maxIdx] + v
                clusterIdx[maxIdx].append(i)
            clusters.append(clusterIdx)
        return clusters