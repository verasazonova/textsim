# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:10:33 2014

@author: vera
"""

from medical import MedicalReviewAbstracts
import argparse
import re
from gensim.models import TfidfModel
from gensim import corpora, matutils
import math
import numpy as np

def calculate_length(doc):
    sum = 0
    for word, weight in doc:
        sum += weight * weight
    if sum > 0:
        return math.sqrt(sum)
    else:
        print doc
    return 0

def calculate_similarity(doc1, doc2, sim_dict):
    # print model[doc1]
    # print model[doc2]
    sum = 0
    if not doc1 or not doc2:
        return 0
    for word_1, weight_1 in doc1:
        for word_2, weight_2 in doc2:
            if word_1 < word_2:
                key = (word_1, word_2)
            else:
                key = (word_2, word_1)
            if key in sim_dict:
                sum += weight_1 * weight_2 * sim_dict[key]
    return sum / (calculate_length(doc1) * calculate_length(doc2))

class SimDictModel:

    def __init__(self, filename, corpus=None, no_below=1, no_above=1):
        self.dictionary = corpora.Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        self.bow_corpus = [self.dictionary.doc2bow(text) for text in corpus]
        self.tfidf_model = TfidfModel(self.bow_corpus, normalize=True)
        if filename is None:
            self.simdict = None
        else:
            self.simdict = self.readdict(filename, self.dictionary)


    @staticmethod
    def readdict(filename, dictionary=None):
        similarity_model = {}
        with open(filename, 'r') as f:
            for line in f:
                # print line
                m = re.search('(.*)<>(.*)\(.*\)<>(.*)\(.*\)', line.strip())
                if m:
                    similarity = float(m.group(1))
                    word1 = str((m.group(2)).replace(" ", "_"))
                    word2 = str((m.group(3)).replace(" ", "_"))
                    if (word1 in dictionary.token2id) and (word2 in dictionary.token2id):
                        # should they be ordered?
                        id1 = dictionary.token2id[word1]
                        id2 = dictionary.token2id[word2]
                        if id1 < id2:
                            similarity_model[(id1, id2)] = similarity
                        else:
                            similarity_model[(id2, id1)] = similarity
                else:
                    print "Parsing error"
                    return None
        return similarity_model


    """ corpus is a bow corpus """
    def calculate_similarities(self, corpus):
        num_terms = len(self.dictionary)
        if self.simdict is None:
            return matutils.corpus2dense(corpus, num_terms=num_terms).T
        similarities = np.array([[calculate_similarity(doc1, doc2, self.simdict) for doc1 in self.bow_corpus] for doc2 in corpus])
        print similarities.shape
        print matutils.corpus2dense(corpus, num_terms=num_terms).shape
        return np.concatenate((matutils.corpus2dense(corpus, num_terms=num_terms).T, similarities), axis=1)


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    arguments = parser.parse_args()

    if arguments.filename is None:
        dataset_filename = "/home/vera/Work/TextVisualization/MedAb_Data/units_Estrogens.txt"
        filename = "/home/vera/Work/TextVisualization/dicts/estrogens-mesh-msr-vector.txt"
    else:
        filename = arguments.filename
        dataset_filename = "none"

    mra = MedicalReviewAbstracts(dataset_filename, ['M'])

#    model, corpus, dictionary = make_tfidf_model(mra)

#    sim_dict = readdict(filename, dictionary)

#    print calculate_similarities(model[corpus], sim_dict, num_terms=len(dictionary)).shape

# print model[corpus[0]]
# for word in word_list:
# if word in dictionary.token2id:



# print dictionary
if __name__ == "__main__":
    __main__()