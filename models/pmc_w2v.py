__author__ = 'verasazonova'

from gensim.models import Word2Vec, TfidfModel
from gensim import corpora, matutils
from os.path import join, basename
from corpus.medical import MedicalReviewAbstracts, word_valid
from corpus.pmc import PubMedCentralOpenSubset
from models.mlda import ModelClassifier
import numpy as np
import os.path
import logging
import argparse
import re
from sklearn.preprocessing import normalize
from sklearn import neighbors
import sklearn.metrics.pairwise as smp


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

    def corpus2dense(self, corpus=None, raw_target=None, single=False):

        logging.info("W2V: with type: %s" % self.model_type)

        target = None
        if single:
            print corpus
            bow_corpus = [self.dictionary.doc2bow(corpus)]
        else:
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
            inc = w2v_length
        if self.model_type == 'stacked_vertically':
            data = np.zeros((n_docs * n_terms, w2v_length))
            inc = 1
            target  = np.zeros(n_docs * n_terms)
        elif self.model_type == 'averaged':
            data = np.zeros((n_docs,  w2v_length))
        elif self.model_type == 'none':
            data = matutils.corpus2dense(self.tfidf_model[bow_corpus], num_terms=len(self.dictionary)).T
            logging.info("W2V: returning tfidf corpus %s " % ( data.shape, ))
            return data
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
                    elif self.model_type == 'stacked_vertically':
                        data[doc_cnt + cnt, :] = weight * self.w2v_model[word]
                        if raw_target is not None:
                            target[doc_cnt + cnt] = raw_target[doc_cnt]
                    else:
                        data[doc_cnt, cnt] = weight
                cnt += inc
            norm = np.linalg.norm(data[doc_cnt])
            if norm != 0:
                data[doc_cnt] /= norm
        logging.info("W2V: returning data with shape %s" % (data.shape, ))
        data = normalize(data)
        if raw_target is not None:
            #print data[0:100, :]
            #target = target[~np.all(data == 0, axis=1)]
            #data = data [~np.all(data == 0, axis=1)]
            #print data[0:100, :]
            logging.info("W2V: returning data with shape %s" % (data.shape, ))
            return data, target
        return data



class W2VModelDistanceClassifier(ModelClassifier):

    def __init__(self, no_below=1, no_above=1, w2v_model=None, model_type='averaged'):
        super(W2VModelDistanceClassifier, self).__init__(no_below=no_below, no_above=no_above, mallet=False, n_topics=0)
        self.w2v_model = w2v_model
        self.model_type = model_type
        logging.info("W2V_distance: initilialized with type")

    def build_models(self, x, model=None):
        logging.info("W2V: building a model")
        self.model = W2VModel(w2v_model=model, corpus=x, no_below=self.no_below, no_above=self.no_above,
                              model_type=self.model_type, topn=0)

        self.clf = neighbors.KNeighborsClassifier(algorithm='auto', metric=smp.cosine_distances)
        logging.info("W2V: model built")

    def pre_process(self, x):
        x_data = self.model.corpus2dense(x)
        logging.info("%s " % (x_data.shape, ))
        return x_data

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

    def pre_process(self, x, y=None, single=False):
        logging.info("W2V: pre-processing data")
        logging.info("W2V: model %s %s" % (self.model, single))
        if self.model_type == 'none' or y is None:
            x_data = self.model.corpus2dense(x, single=single)
            logging.info("%s " % (x_data.shape, ))
            return x_data
        else:
            x_data, y_data = self.model.corpus2dense(x, y, single=single)
            logging.info("%s %s" % (x_data.shape, y_data.shape))
            return x_data, y_data

    def fit(self, x, y, model=None):
        self.build_models(x, model)
        self.classes_, self.indeces_ = np.unique(y, return_index=True)
        logging.info("MC: model %s, classifier %s" % (model, self.clf))
        if self.model_type == 'none':
            x_data = self.pre_process(x)
            y_data = y
        else:
            x_data, y_data = self.pre_process(x, y)
        logging.info("ModelClassifier: fitting data with shape %s " % (x_data.shape,))
        return self.clf.fit(x_data, y_data)

    def predict(self, x_array):
        y = []
        for x in x_array:
            x_data = self.pre_process(x, single=True)
            predictions =  self.clf.predict(x_data)
            if self.model_type == 'none':
                y.append(predictions)
            else:
                ## x here is not necessarily single!!!
                logging.info("Predict: %s" % (len(predictions), ))
                y.append( np.argmax( np.bincount(map(int, predictions)) ))
        return y

def create_w2v_model(filename, size=100, window=5):
    pmc_corpus = PubMedCentralOpenSubset(filename)
    model = Word2Vec(pmc_corpus, size=size, window=window, workers=4)
    model.save(join(basename(filename), "pmc_%i_%i" % (size, window)))

sim_cache = {}


def augment_corpus(corpus=None, w2v_model=None, topn=[100], perword=False):
    """
    Local parameter controls whether the similar words are taken per word or with the respect to the whole document
    """
    max_topn = np.max(np.array(topn))
    print max_topn
    augmented_corpus = {}
    for n in topn:
        augmented_corpus[n] = []
    for i, text in enumerate(corpus):
        words_in_model = [word for word in text if word in w2v_model]
        augmented_text = {}
        for n in topn:
            augmented_text[n] = text[:]
        if perword:
            sim_words = []
            for word in words_in_model:
                if word not in sim_cache:
                    sim_cache[word] = [re.sub(r"\W", "_", tup[0])
                                       for tup in w2v_model.most_similar(positive=word, topn=max_topn)
                                       if word_valid(tup[0])]
                sim_words = sim_cache[word]
                for n in topn:
                    augmented_text[n] += sim_words[0:n]
        else:
            sim_words = [re.sub(r"\W", "_", tup[0]) for tup in w2v_model.most_similar(positive=words_in_model, topn=max_topn)
                         if word_valid(tup[0])]
            for n in topn:
                augmented_text[n] += sim_words[0:n]

        for n in topn:
            augmented_corpus[n].append(augmented_text[n])
    if len(topn) == 1:
        print "Returning a single corpus"
        return augmented_corpus[topn[0]]
    else:
        return augmented_corpus


def prep_arguments(arguments):

    prefix = os.environ.get("MEDAB_DATA")
    datasets = []
    filenames = []
    if arguments.dataset is None:
        datasets = ["Estrogens"]
        filenames = [prefix + "/units_Estrogens.txt"]
    else:
        datasets = arguments.dataset
        print datasets, prefix
        filenames =  [prefix + "/units_" + dataset + ".txt" for dataset in datasets]

    topn = map(int, arguments.topn)
    perword = arguments.perword
    return datasets, filenames, topn, perword


def write_augmented_corpus(dataset, filename, topn, w2v_model):
    mra = MedicalReviewAbstracts(filename, ['T', 'A'])
    x = np.array([text for text in mra])
    y = mra.get_target()
    augmented = augment_corpus(corpus=x, w2v_model=w2v_model, topn=topn, perword=True)
    for n in topn:
        with open(str(dataset) + "-" + str(n) + "pw.txt", 'w') as fout:
            for text, c in zip(augmented[n], y):
                fout.write(" ".join(text) + ", " + str(c) + "\n")

def __main__():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store', nargs="+", dest='dataset', help='Dataset name')
    parser.add_argument('-m', action='store', dest='model', help='model')
    parser.add_argument('--topn', action='store', nargs="+", dest='topn', default='0', help='Dataset name')
    parser.add_argument('--pword', action='store_true', dest='perword', help='whether similar words taken per word')
    arguments = parser.parse_args()

    datasets, filenames, topn, perword = prep_arguments(arguments)

    w2v_model = Word2Vec.load(arguments.model)
    w2v_model.init_sims(replace=True)

    #for dataset, filename in zip(datasets, filenames):
        #write_augmented_corpus(dataset, filename, topn, w2v_model)




if __name__ == "__main__":
    __main__()