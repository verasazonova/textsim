#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:22:19 2014

@author: vera
"""

import argparse
import os.path
import re
#from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


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
if os.path.isfile(stop_filename):
    stop = set([word.strip() for word in open(stop_filename, 'r').readlines()])
else:
#    stop = set(stopwords.words('english'))
    stop = set(['a', 'the'])

def word_valid(word):
    if (word in stop) or len(word) < 2 or re.match('\d+([,.]\d)*', word):
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


class MedicalReviewAbstracts:
    def __init__(self, filename, article_fields):
        self.articles = readarticles(filename, article_fields)

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


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    arguments = parser.parse_args()

    if arguments.filename is None:
        filename = "/home/vera/Work/TextVisualization/MedAb_Data/units_Estrogens.txt"
    else:
        filename = arguments.filename

    print filename
    mra = MedicalReviewAbstracts(filename, ['T', 'A', 'M'])

    # print mra.get_target()
    for article in mra:
        print article


if __name__ == "__main__":
    __main__()