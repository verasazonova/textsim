__author__ = 'verasazonova'

from nltk.corpus import reuters
import argparse
import numpy as np
from corpus.medical import word_valid

class ReutersDataset():

    def __init__(self, categories=None, lower=True):
        if categories == None or len(categories) == 1:
            self.fileids = reuters.fileids()
        else:
            self.fileids = reuters.fileids(categories)
        self.categories = categories

        self.lower = lower


    def get_subset(self, fileid):
        if self.lower:
            return [ word.lower() for word in reuters.words(fileid) if word_valid(word) ]
        else:
            return [ word for word in reuters.words(fileid) if word_valid(word) ]

    def __iter__(self):
        for fileid in self.fileids:
            yield self.get_subset(fileid)


    def get_train(self):
        x = [ self.get_subset(fileid) for fileid in self.fileids if fileid.startswith("train")]
        y = [ 1 if self.categories[0] in reuters.categories(fileid) else 0
                       for fileid in self.fileids if fileid.startswith("train")]
        return x, y

    def get_test(self):
        x = [ self.get_subset(fileid) for fileid in self.fileids if fileid.startswith("test")]
        y = [ 1 if self.categories[0] in reuters.categories(fileid) else 0
                       for fileid in self.fileids if fileid.startswith("test")]
        return x, y


    def get_target(self):

        # cat1 vs. cat2
        if len(self.categories) > 1:
            target = [ [cat for cat in reuters.categories(fileid) if cat in self.categories][0]
                       for fileid in self.fileids]
        # cat1 vs. not cat1
        else:
            target = [ 1 if self.categories[0] in reuters.categories(fileid) else 0
                       for fileid in self.fileids]
        self.classes, target = np.unique(target, return_inverse=True)
        return target




def explore_categories(max_len=5000, min_len=100, percentage=0.3):
    for cat in reuters.categories():
        for cat2 in reuters.categories():
            if cat2 > cat:
                if  len(set(reuters.fileids(cat)) & set(reuters.fileids(cat2))) == 0:
                    l1 = len(reuters.fileids(cat))
                    l2 = len(reuters.fileids(cat2))
                    if ( (l1 + l2) > min_len) and ( (l1 + l2) < max_len) and float((min(l1, l2))/float(l1+l2) > percentage):
                        print cat, cat2, l1 + l2, float(min(l1, l2))/float(l1+l2)

def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', action='store', nargs='+', dest='categories', help='Data filename')
    parser.add_argument('-min', action='store', dest='min', help='Data filename')
    parser.add_argument('-max', action='store', dest='max', help='Data filename')
    parser.add_argument('-p', action='store', dest='percentage', help='Data filename')
    arguments = parser.parse_args()

    rd = ReutersDataset(arguments.categories)
    x, y = rd.get_test()
    print len(x)
    print len(y)
    x, y = rd.get_train()
    print len(x)
    print len(y)
    print y
    #print rd.get_train()
    #print arguments.percentage
    #explore_categories(max_len=int(arguments.max), min_len=int(arguments.min), percentage=float(arguments.percentage))

if __name__ == "__main__":
    __main__()