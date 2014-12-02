#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:22:19 2014

@author: vera
"""

import argparse
import os.path
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gensim import corpora
from gensim.models.doc2vec import LabeledSentence

def readarticles(filename, article_fields):
    article_list = []
    article_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            # print line
            m = re.search('\*\*\*\*\*\* (.+)', line.strip())
            if m:
                if not (article_dict == {}):
                    article_list.append(article_dict)
                    # reinitialize the out string
                article_dict = {'id': m.group(1)}
            else:
                m = re.search('----([K|T|A|P|M]) (.+)', line.strip())
                if m:
                    # if one of the valid options
                    if m.group(1) == 'K':
                        article_dict['class'] = m.group(2)
                    if m.group(1) in article_fields:
                        # if in the middle of the article, and the fields are one of the valid ones -
                        # create the out dictionary
                        for field in article_fields:
                            if m.group(1) == field:
                                # for separate fields - copy the field
                                str_field = m.group(2)
                                article_dict[field] = str_field.replace("\"", "\\\"")

        # process the write out string for the last article
        article_list.append(article_dict)
    return article_list


stop_filename = "stopwords.txt"
stop_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), stop_filename)
print stop_path
if os.path.isfile(stop_path):
    print "Using stopwords.txt as stopword list"
    stop = set([word.strip() for word in open(stop_path, 'r').readlines()])
else:
    print "Using nltk stopwords"
    stop = set(stopwords.words('english'))
#    stop = set(['a', 'the'])

def word_valid(word):
    if (word in stop) or len(word) < 2 or re.match('\d+([,.]\d)*', word) or re.match(r".*\\.*", word) \
            or re.match(r"\W+", word):
        return False
    return True


def word_tokenize(text):
    tokens = [word.translate(None, '!?.,;:\'\"') for word in text.translate(None, '()[]').lower().split() if
              word_valid(word)]
    return tokens


#def sent_tokenize(text):
#    tokens = [sent for sent in text.split('. ')]
#    tokens = nl
#    return tokens


def mesh_tokenize(text):
    tokens = []
    for word in text.lower().split():
        m = re.match("([sm]_)+(.*)_mesh", word)
        if m:
            tokens.append(m.group(2))
    return tokens


class AugmentedCorpus:
    def __init__(self, filename):
        self.filename = filename
        self.target = []

    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                self.target.append(int(line.strip().split(',')[1]))
                yield line.strip().split(',')[0].split()

    def get_target(self):
        if self.target:
            return self.target
        else:
            print "you must iterated the class first"

class MedicalReviewAbstracts:
    def __init__(self, filename, article_fields):
        self.articles = readarticles(filename, article_fields)
        self.dataset = os.path.basename(filename)

    def __iter__(self):
        for article in self.articles:
            text_tokens = []
            mesh_tokens = []
            if ('T' in article) and ('A' in article):
                text_tokens = word_tokenize(article['T'] + article['A'])
            elif 'A' in article:
                text_tokens = word_tokenize(article['A'])
            elif 'T' in article:
                text_tokens = word_tokenize(article['T'])

            if 'M' in article:
                mesh_tokens = mesh_tokenize(article['M'])
            yield text_tokens + mesh_tokens

    def get_target(self):
        return [1 if article['class'] == 'I' else 0 for article in self.articles]


    def print_statistics(self):

        n = len(self.articles)
        pos_ind = [i for i in range(n) if self.articles[i]['class'] == 'I' ]
        n_pos = len(pos_ind)
        #print "Dataset, Percent positives, # positives, # total: "
        return self.dataset, n_pos*100.0 / n, n_pos, n



class LabeledMedicalReviewAbstracts(MedicalReviewAbstracts):

    def __iter__(self):
        for article in self.articles:
            text_tokens = []
            mesh_tokens = []
            if ('T' in article) and ('A' in article):
                text_tokens = word_tokenize(article['T'] + article['A'])
            elif 'A' in article:
                text_tokens = word_tokenize(article['A'])
            elif 'T' in article:
                text_tokens = word_tokenize(article['T'])

            if 'M' in article:
                mesh_tokens = mesh_tokenize(article['M'])

            yield LabeledSentence( text_tokens + mesh_tokens, [article['id']] )



def print_stats(mra):
    name, p_pos, n_pos, n = mra.print_statistics()
    x = [article for article in mra]
    y = mra.get_target()
    dictionary = corpora.Dictionary(x)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    n_w =  len(dictionary)
    x_pos = [article for i, article in enumerate(mra) if y[i] == 1]
    dictionary_pos = corpora.Dictionary(x_pos)
    dictionary_pos.filter_extremes(no_below=2, no_above=0.9)
    n_w_pos = len(dictionary_pos)
    print ", ".join(map(str, [name, p_pos, n_pos, n, n_w, n_w_pos]))


def get_filename(dataset):
    prefix = os.environ.get("MEDAB_DATA")
    return prefix + "/units_" + dataset + ".txt"

def prep_arguments(arguments):

    prefix = os.environ.get("MEDAB_DATA")
    datasets = []
    filenames = []
    if (arguments.filename is None) and (arguments.dataset is None):
        datasets = ["Estrogens"]
        filenames = [prefix + "/units_Estrogens.txt"]
    elif arguments.filename is None:
        datasets = arguments.dataset
        print datasets, prefix
        filenames =  [prefix + "/units_" + dataset + ".txt" for dataset in datasets]
    else:
        exit()

    return datasets, filenames


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', nargs='+', dest='filename', help='Data filename')
    parser.add_argument('-d', action='store', nargs="+", dest='dataset', help='Dataset name')
    arguments = parser.parse_args()

    datasets, filenames = prep_arguments(arguments)
    for filename in filenames:
#        print filename
        mra = LabeledMedicalReviewAbstracts(filename, ['T', 'A'])
#        print_stats(mra)
        for article in mra:
            print article

if __name__ == "__main__":
    __main__()