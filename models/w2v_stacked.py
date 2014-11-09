__author__ = 'verasazonova'

import models.mlda
import logging
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model, svm

class BOWClassifier(models.mlda.ModelClassifier):
    def __init__(self, no_below=2, no_above=0.9):
        super(BOWClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        logging.info("BOW classifier: initilialized with type ")



class W2VStackedClassifier(models.mlda.ModelClassifier):

    def __init__(self, no_below=2, no_above=0.9, n_components=100, logistic_C=6000, learning_rate=0.06, use_svm=False):
        super(W2VStackedClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        self.logistic = linear_model.LogisticRegression()

        self.rbm = BernoulliRBM(random_state=0, verbose=True)
        self.rbm.learning_rate = learning_rate
        self.rbm.n_iter = 20
        # More components tend to give better prediction performance, but larger
        # fitting time
        self.rbm.n_components = n_components
        self.logistic.C = logistic_C
        self.use_svm = use_svm

        if not self.use_svm:
            self.clf = Pipeline(steps=[('rbm', self.rbm), ('logistic', self.logistic)])
        else:
            self.clf = svm.SVC(kernel='linear', C=1)

        logging.info("W2V Stacked Classifier: initilialized with clf %s " % (self.clf))

    def get_params(self, deep=True):        # suppose this estimator has parameters "alpha" and "recursive"
        return {"no_below": self.no_below, "no_above": self.no_above, "n_components": self.rbm.n_components,
                "logistic_C": self.logistic.C, "learning_rate": self.rbm.learning_rate, "use_svm": self.use_svm}

    #def set_params(self, **parameters):
    #    for parameter, value in parameters.items():
    #        self.setattr(parameter, value)

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

