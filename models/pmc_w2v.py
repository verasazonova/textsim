__author__ = 'verasazonova'

from gensim.models import Word2Vec, TfidfModel
from gensim import corpora, matutils
from os.path import join, basename
from corpus.medical import MedicalReviewAbstracts
from corpus.pmc import PubMedCentralOpenSubset
from models.mlda import ModelClassifier
import numpy as np
import logging
from sklearn.preprocessing import normalize


class W2VModel():
    def __init__(self, w2v_model=None, corpus=None, no_below=1, no_above=1.0, model_type='averaged', topn=100):
        self.dictionary = corpora.Dictionary(corpus)
        self.no_below = no_below
        self.no_above = no_above
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]
        self.tfidf_model = TfidfModel(self.bow_corpus, normalize=True)
        self.topn = topn
        self.model_type = model_type

        if self.model_type == 'none':
            self.w2v_model = None
        else:
            self.w2v_model = w2v_model

        if self.model_type == 'augmented':
            if self.topn == 0:
                self.model_type = 'none'
            else:
                logging.info("W2V augmented model with topn = %i" % self.topn)
                augmented_corpus = self.augment_corpus(corpus)
                self.augmented_dict = corpora.Dictionary(augmented_corpus)
                self.augmented_dict.filter_extremes(no_below=self.no_below, no_above=self.no_above)
                self.augmented_bow = [self.augmented_dict.doc2bow(text) for text in augmented_corpus]
                self.augmented_tfidf = TfidfModel(self.augmented_bow, normalize=True)

    def augment_corpus(self, corpus=None):
        augmented_corpus = []
        for text in corpus:
            words_in_model = [word for word in text if word in self.w2v_model]
            sim_words = [tup[0] for tup in self.w2v_model.most_similar(positive=words_in_model, topn=self.topn)]
            augmented_corpus.append(text[:] + sim_words)
        return augmented_corpus

    def corpus2dense(self, corpus=None):

        logging.info("W2V: with type: %s" % self.model_type)

        bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]

        if self.w2v_model is None:
            w2v_length = 0
            logging.info("W2V: model is none, with type %s " % self.model_type)
        else:
            w2v_length = len(self.w2v_model['test'])
        n_terms = len(self.dictionary)
        n_docs = len(bow_corpus)
        inc = 0

        if self.model_type == 'stacked':
            data = np.zeros((n_docs, n_terms * w2v_length))
        elif self.model_type == 'averaged':
            data = np.zeros((n_docs,  w2v_length))
            inc = w2v_length
        elif self.model_type == 'none':
            return matutils.corpus2dense(self.tfidf_model[bow_corpus], num_terms=len(self.dictionary)).T
        elif self.model_type == 'augmented':
            augmented_bow = [self.augmented_dict.doc2bow(text) for text in self.augment_corpus(corpus)]
            return matutils.corpus2dense(self.augmented_tfidf[augmented_bow], num_terms=len(self.augmented_dict)).T

        # Need to check whether dictionary contains words or ids
        # Need to multiply by the appropriate weights: boolean or tfidf

        for doc_cnt, doc in enumerate(self.tfidf_model[bow_corpus]):
            cnt = 0
            for word_id, weight in doc:
                word = self.dictionary[word_id]
                if word in self.w2v_model:
                    if self.model_type == 'averaged':
                        data[doc_cnt] += weight * self.w2v_model[word] / n_terms
                    elif self.model_type == 'stacked':
                        data[doc_cnt, cnt:cnt + w2v_length] = weight * self.w2v_model[word]
                    else:
                        data[doc_cnt, cnt] = weight
                cnt += inc
            norm = np.linalg.norm(data[doc_cnt])
            if norm != 0:
                data[doc_cnt] /= norm
        logging.info("W2V: returning data with shape %s" % (data.shape, ))
        data = normalize(data)
        return data


class W2VModelClassifier(ModelClassifier):
    """
    Builds a join tfidf - similarity matrix model.
    """

    def __init__(self, no_below=1, no_above=1, w2v_model=None, model_type='averaged', topn=100):
        super(W2VModelClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        self.w2v_model = w2v_model
        self.model_type = model_type
        self.topn = topn
        logging.info("W2V: initilialized with type %s " % model_type)

    def build_models(self, x, model=None):
        logging.info("W2V: building a model")
        self.model = W2VModel(w2v_model=model, corpus=x, no_below=self.no_below, no_above=self.no_above,
                              model_type=self.model_type, topn=self.topn)
        logging.info("W2V: model built")

    def pre_process(self, x):

        logging.info("W2V: pre-processing data")
        logging.info("W2V: model %s" % (self.w2v_model,))
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
#    w2v_model = None
    mra = MedicalReviewAbstracts(filename, ['T'])
    x = np.array([text for text in mra])
    stacked_model = W2VModel(w2v_model, corpus=x, no_above=0.9, no_below=2, model_type='stacked')
    print stacked_model.corpus2dense()

if __name__ == "__main__":
    __main__()