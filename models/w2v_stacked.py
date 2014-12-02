__author__ = 'verasazonova'

import models.mlda
import logging
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm
from sklearn.base import BaseEstimator, ClassifierMixin
from gensim import corpora, models, matutils


class BaseClassifier(BaseEstimator, ClassifierMixin):
#    __metaclass__ = abc.ABCMeta

    def __init__(self, clf=None):
        if clf is None:
            self.clf = svm.SVC(kernel='linear', C=1)
        else:
            self.clf = clf

    def build_models(self, x, model=None):
        raise NotImplementedError("Please Implement this method")

    def pre_process(self, x):
        raise NotImplementedError("Please Implement this method")

    def fit(self, x, y, model=None):
        self.build_models(x, model)
        logging.info("BC: model %s, classifier %s" % (model, self.clf))
        x_data = self.pre_process(x)
        logging.info("BC: fitting data with shape %s " % (x_data.shape,))
        return self.clf.fit(x_data, y)

    def decision_function(self, x):
        return self.clf.decision_function(self.pre_process(x))

    def predict(self, x):
        x_data = self.pre_process(x)
        return self.clf.predict(x_data)

    def predict_proba(self, x):
        x_data = self.pre_process(x)
        return self.clf.predict_proba(x_data)


class BOWClassifier(BaseClassifier):
    def __init__(self, no_below=2, no_above=0.9, clf=None):
        super(BOWClassifier, self).__init__(clf)
        self.no_below = no_below
        self.no_above = no_above
        logging.info("BOW classifier: initialized with no_below %s and no_above %s " % (self.no_below, self.no_above))


    def build_models(self, x, model=None):
        self.dictionary = corpora.Dictionary(x)
        self.dictionary.filter_extremes(no_above=self.no_above, no_below=self.no_below)
        self.bow = [self.dictionary.doc2bow(text) for text in x]
        self.tfidf = models.TfidfModel(self.bow, dictionary=self.dictionary, normalize=True)

    def pre_process(self, x):
        x_tfidf = self.tfidf[[self.dictionary.doc2bow(text) for text in x]]
        x_data = matutils.corpus2dense(x_tfidf, num_terms=len(self.dictionary)).T
        return x_data


class W2VStackedClassifier(BaseClassifier):

    def __init__(self, learning_rate=0.06, n_components=100, logistic_C=6000):
        self.logistic = linear_model.LogisticRegression()
        print learning_rate, n_components
        self.rbm = BernoulliRBM(random_state=0, verbose=True)
        self.rbm.learning_rate = learning_rate
        self.rbm.n_iter = 20
        # More components tend to give better prediction performance, but larger
        # fitting time
        self.rbm.n_components = n_components
        self.logistic.C = logistic_C
        self.clf = Pipeline(steps=[('rbm', self.rbm), ('logistic', self.logistic)])
        logging.info("W2v clf %s " % (self.clf))


    def build_models(self, x, model=None):
        logging.info("W2V: building a model")
        self.max_textlen = max( [len(text) for text in x])
        logging.info("Max text length is %s " % (self.max_textlen, ))
        self.w2v_model = model

    def pre_process(self, x):
        logging.info("W2V Stacked: pre-processing data of shape %s " % (x.shape, ))
        n_docs = len(x)
        w2v_length = len(self.w2v_model['test'])
        logging.info("W2V Stacked: max_textlen %s, w2v-len %s " % (self.max_textlen, w2v_length))
        data = np.zeros( (n_docs, self.max_textlen * w2v_length))
        for doc_cnt, text in enumerate(x):
            cnt = 0
            for word in text:
                if word in self.w2v_model:
                    data[doc_cnt, cnt:cnt + w2v_length] = self.w2v_model[word]
                cnt += w2v_length
                if cnt >= self.max_textlen * w2v_length:
                    break
        logging.info("W2V Stacked: returning pre-processed data of shape %s" % (data.shape, ))
        return data


