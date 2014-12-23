__author__ = 'verasazonova'


import xml.etree.ElementTree as ET
import re
from os import listdir
from os.path import isfile, isdir, join
import codecs
from corpus.medical import sent_tokenize, word_tokenize
from gensim.models.doc2vec import LabeledSentence

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


