__author__ = 'verasazonova'

import logging
import argparse
from gensim.models import Doc2Vec, Word2Vec
from os.path import isdir, dirname, join
from corpus.pmc import PubMedCentralOpenSubset
from corpus.medical import MedicalReviewAbstracts


def create_vector_model(corpus, output_dir=None, size=100, window=8, d2v=False, train_docs=False, w2v_model=None):
    logging.info("Corpus initialized")
    if d2v:
        model = Doc2Vec(corpus, size=size, alpha=0.025, window=window, min_count=5, sample=0, seed=1,
                                        workers=4, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0,
                                        train_words=True, train_lbls=train_docs, initial=w2v_model)
        name = "doc2vector_models/pmc_d2v_%i_%i" % (size, window)

    else:
        model = Word2Vec(corpus, size=size, window=window, workers=4)
        name = "word2vector_models/pmc_%i_%i" % (size, window)


    if isdir(dirname(output_dir)):
        model_filename = join(dirname(output_dir), name)
    else:
        model_filename = str(name)

    logging.info("Model created")

    if d2v:
        model.save_word2vec_format(model_filename, binary=True)
    else:
        model.save_word2vec_format(model_filename, binary=True)
    logging.info("Model saved ins %s" % model_filename)



def convert_format(filename):
    model = Word2Vec.load(filename)
    model.save_word2vec_format(filename, binary=True)

def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', action='store', dest='window', help='Window')
    parser.add_argument('-s', action='store', dest='size', help='Size')
    parser.add_argument('--d2v', action='store_true', dest='d2v', help='Size')
    parser.add_argument('-f', action='store', dest='filename', help='Filename of the corpus')
    parser.add_argument('--type', action='store', dest='type', default='pmc', help='Corpus type: pmc or medab')
    parser.add_argument('--model', action='store', dest='w2v_model', default='None', help='W2V model to initialize')

    arguments = parser.parse_args()

    model_path = "/Users/verasazonova/no-backup/"


    if arguments.type == 'pmc':
        size = int(arguments.size)
        window = int(arguments.window)
        corpus_path = "/Users/verasazonova/no-backup/pubmed_central/"
        filename = join(corpus_path, arguments.filename)
        corpus = PubMedCentralOpenSubset(filename, labeled=arguments.d2v)
        train_docs = False # non need to train docs in the unlabeled dataset

    elif arguments.type == 'medab':
        size = int(arguments.size)
        window = int(arguments.window)
        corpus_path = "/Users/verasazonova/no-backup/medab_data/"
        filename = join(corpus_path, arguments.filename)
        corpus = MedicalReviewAbstracts(filename, ['T', 'A'], labeled=arguments.d2v, tokenize=True)
        train_docs = True # train docs, as we will use these later.

    else:
        raise Exception("Invalid corpus type")


    logging.info("creating model with size %s and window %s" % (size, window))
    create_vector_model(corpus, output_dir=model_path, size=size, window=window, d2v=arguments.d2v,
                        train_docs=train_docs, w2v_model=arguments.w2v_model)


if __name__ == "__main__":
    __main__()