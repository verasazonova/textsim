__author__ = 'verasazonova'


import argparse
import xml.etree.ElementTree as ET
import re
from os import listdir
from os.path import isfile, isdir, join, dirname
import codecs
from corpus.medical import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
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
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                sents = sent_tokenize(line)
                for sent in sents:
                    current_sent = word_tokenize(sent)
                    if current_sent:
                        yield current_sent


def create_w2v_model(filename, size=100, window=5):
    pmc_corpus = PubMedCentralOpenSubset(filename)
    logging.info("Corpus initialized")
    model = Word2Vec(pmc_corpus, size=size, window=window, workers=4)
    logging.info("Model created")
    if isdir(dirname(filename)):
        model_filename = join(dirname(filename), "pmc_%i_%i" % (size, window))
    else:
        model_filename = str("pmc_%i_%i" % (size, window))
    model.save(model_filename)
    logging.info("Model saved ins %s" % model_filename)


def __main__():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-w', action='store', dest='window', help='Window')
    parser.add_argument('-s', action='store', dest='size', help='Size')

    arguments = parser.parse_args()

    path = "/Users/verasazonova/no-backup/pubmed_central/"
    #construct_pmc(path)

    filename = join(path, "pmc_corpus_TA.txt")
    size = int(arguments.size)
    window = int(arguments.window)
    logging.info("creating model with size %s and window %s" % (size, window))
    create_w2v_model(filename, size=size, window=window)


if __name__ == "__main__":
    __main__()