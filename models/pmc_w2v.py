__author__ = 'verasazonova'

from gensim.models import Word2Vec, TfidfModel
from gensim import corpora, matutils
from os.path import join, basename
from corpus.medical import MedicalReviewAbstracts
from corpus.pmc import PubMedCentralOpenSubset
from models.mlda import ModelClassifier
import numpy as np
import logging


class W2VModel():
    def __init__(self, w2v_model=None, corpus=None, no_below=1, no_above=1.0, model_type='averaged'):
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]
        self.tfidf_model = TfidfModel(self.bow_corpus, normalize=True)
        self.model_type = model_type
        self.w2v_model = w2v_model

    def corpus2dense(self, corpus=None):

        logging.info("W2V: with type: %s" % self.model_type)

        bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]

        if self.w2v_model is None:
            w2v_length = 0
        else:
            w2v_length = len(self.w2v_model['test'])
        n_terms = len(self.dictionary)
        n_docs = len(bow_corpus)
        inc = 0

        if self.model_type == 'stacked':
            data = np.zeros((n_docs, n_terms * w2v_length))
        elif self.model_type == 'averaged':
            data = np.empty((n_docs,  w2v_length))
            inc = w2v_length
        elif self.model_type == 'none':
            self.w2v_model = None

        # Need to check whether dictionary contains words or ids
        # Need to multiply by the appropriate weights: boolean or tfidf
        if self.w2v_model is None:
            return matutils.corpus2dense(self.tfidf_model[bow_corpus], num_terms=len(self.dictionary)).T

        for doc_cnt, doc in enumerate(self.tfidf_model[bow_corpus]):
            cnt = 0
            for word_id, weight in doc:
                word = self.dictionary[word_id]
                if word in self.w2v_model:
                    if self.model_type == 'averaged':
                        data[doc_cnt] += weight * self.w2v_model[word]
                    elif self.model_type == 'stacked':
                        data[doc_cnt, cnt:cnt + w2v_length] = weight * self.w2v_model[word]
                    else:
                        data[doc_cnt, cnt] = weight
                cnt += inc
        logging.info("W2V: returning data with shape %s" % (data.shape, ))
        return data / n_terms


class W2VModelClassifier(ModelClassifier):
    """
    Builds a join tfidf - similarity matrix model.
    """

    def __init__(self, no_below=1, no_above=1, w2v_model=None, model_type='averaged'):
        super(W2VModelClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        self.w2v_model = w2v_model
        self.model_type = model_type
        logging.info("W2V: initilialized")

    def build_models(self, x):
        logging.info("W2V: building a model")
        self.model = W2VModel(w2v_model=self.w2v_model, corpus=x, no_below=self.no_below, no_above=self.no_above, model_type=self.model_type)
        logging.info("W2V: model built")

    def pre_process(self, x):

        logging.info("W2V: pre-processing data")
        x_data = self.model.corpus2dense(x)
        return x_data


def create_w2v_model(filename, size=100, window=5):
    pmc_corpus = PubMedCentralOpenSubset(filename)
    model = Word2Vec(pmc_corpus, size=size, window=window, workers=4)
    model.save(join(basename(filename), "pmc_%i_%i" % (size, window)))


def __main__():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    path = "/Users/verasazonova/no-backup/pubmed_central/"
    name = "pmc_100_5"
    filename = "/Users/verasazonova/no-backup/medab_data/units_Estrogens.txt"
    # create_w2v_model(join(path, name))

    w2v_model = Word2Vec.load(join(path, name))

    mra = MedicalReviewAbstracts(filename, ['T'])
    x = np.array([text for text in mra])
    stacked_model = W2VModel(w2v_model, corpus=x, no_above=0.9, no_below=2, model_type='stacked')
    print stacked_model.corpus2dense()

if __name__ == "__main__":
    __main__()