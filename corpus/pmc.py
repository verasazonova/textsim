__author__ = 'verasazonova'


import argparse
import xml.etree.ElementTree as ET
import re
from os import listdir
from os.path import isfile, isdir, join, dirname
import codecs
from corpus.medical import sent_tokenize, word_tokenize
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import logging

def remove_latex(s):
    return re.sub('\\\(.*)(\[.*\])*(\{.*\})*', '', s).strip()


def process_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    file_str = ""
    accepted_tags = set(['article-title', 'astract'])

    for elem in root.iter('*'):
        if elem.tag in accepted_tags: #== 'article-title' or elem.tag == 'abstract': # or elem.tag == 'body':
            for textnode in elem.itertext():
                file_str += textnode + ".  "

    return remove_latex(file_str)


def construct_pmc(path):
    with codecs.open(join(path, "pmc_corpus_TA.txt"), encoding='utf-8', mode='w+') as f:
        for part in listdir(path):
            part_path = join(path, part)
            if isdir(part_path):
                for journal in listdir(part_path):
                    journal_path = join(part_path, journal)
                    if isdir(journal_path):
                        for filename in listdir(journal_path):
                            file_path = join(journal_path, filename)
                            if isfile(file_path) and filename.endswith(".nxml"):
                                f.write(process_file(file_path) + '\n')


class PubMedCentralOpenSubset():
    def __init__(self, filename, labeled=True):
        self.filename = filename
        self.labeled = labeled

    def __iter__(self):
        cnt = 0
        with open(self.filename) as f:
            for line in f:
                sents = sent_tokenize(line)
                for sent in sents:
                    current_sent = word_tokenize(sent)
                    if current_sent:
                        cnt += 1
                        if self.labeled:
                            yield LabeledSentence(current_sent, [str(cnt)])
                        else:
                            yield current_sent


def create_w2v_model(filename, size=100, window=8, d2v=False):
    pmc_corpus = PubMedCentralOpenSubset(filename)
    logging.info("Corpus initialized")
    if d2v:
        model = Doc2Vec(pmc_corpus, size=size, alpha=0.025, window=window, min_count=5, sample=0, seed=1,
                                        workers=4, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0,
                                        train_words=True, train_lbls=False)
        name = "pmc_d2v_%i_%i" % (size, window)
    else:
        model = Word2Vec(pmc_corpus, size=size, window=window, workers=4)
        name = "pmc_%i_%i" % (size, window)

    logging.info("Model created")
    if isdir(dirname(filename)):
            model_filename = join(dirname(filename), name)
    else:
        model_filename = str(name)

    model.save(model_filename)
    logging.info("Model saved ins %s" % model_filename)


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', action='store', dest='window', help='Window')
    parser.add_argument('-s', action='store', dest='size', help='Size')
    parser.add_argument('--d2v', action='store_true', dest='d2v', help='Size')

    arguments = parser.parse_args()

    path = "/Users/verasazonova/no-backup/pubmed_central/"
    #construct_pmc(path)

    filename = join(path, "pmc_corpus_TA.txt")
    size = int(arguments.size)
    window = int(arguments.window)
    logging.info("creating model with size %s and window %s" % (size, window))
    create_w2v_model(filename, size=size, window=window, d2v=arguments.d2v)


if __name__ == "__main__":
    __main__()