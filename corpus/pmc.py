__author__ = 'verasazonova'


import argparse
import xml.etree.ElementTree as ET
import re
from os import listdir
from os.path import isfile, isdir, join
import codecs
from corpus.medical import sent_tokenize, word_tokenize


def remove_latex(s):
    return re.sub('\\\(.*)(\[.*\])*(\{.*\})*', '', s).strip()


def process_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    file_str = ""

    for elem in root.iter('*'):
        if elem.tag == 'article-title' or elem.tag == 'abstract' or elem.tag == 'body':
            for textnode in elem.itertext():
                file_str += textnode + ".  "

    return remove_latex(file_str)


def construct_pmc(path):
    with codecs.open(join(path, "pmc_corpus.txt"), encoding='utf-8', mode='w+') as f:
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
                words = [word_tokenize(sent) for sent in sents]
                yield words


def __main__():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')

    path = "/Users/verasazonova/Work/pubmed_central/"
    construct_pmc(path)



if __name__ == "__main__":
    __main__()