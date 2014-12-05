__author__ = 'verasazonova'

import csv
import numpy as np
import argparse
import re
import sklearn.naive_bayes
import sklearn.utils
#from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
#from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer
from ttp import ttp
#import matplotlib.pyplot as plt

def read_file_annotated(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        y = []
        for row in reader:
            x.append(row[2].decode('utf-8'))
            y.append(row[3])
    return x, y


def read_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        labels = []
        for row in reader:
            str = unicode(row[1], errors='replace')
            x.append(str)
            str = unicode(row[15], errors='replace')
            labels.append(str.replace(' ', '_').replace(',', ',_'))
    return x, labels


class MyParser(ttp.Parser):

    # User defined formatters --------------------------------------------------
    def format_tag(self, tag, text):
        '''Return formatted HTML for a hashtag.'''
        return text.encode('utf-8')

    def format_username(self, at_char, user):
        '''Return formatted HTML for a username.'''
        return user
#        return '<a href="https://twitter.com/%s">%s%s</a>' \
 #              % (user, at_char, user)

    def format_list(self, at_char, user, list_name):
        '''Return formatted HTML for a list.'''
        return '<a href="https://twitter.com/%s/%s">%s%s/%s</a>' \
               % (user, list_name, at_char, user, list_name)

    def format_url(self, url, text):
        '''Return formatted HTML for a url.'''
        return ""
#        return '<a href="%s">%s</a>' % (escape(url), text)


class KenyanTweets():
    def __init__(self, filename):
        self.data, self.labels = read_file(filename)

    def __iter__(self):
        p = MyParser()
        for text in self.data:
            result = p.parse(text)
            yield re.sub(r'([,.?!])(?![\s.,!?])', r'\1 ', result.html)

    def get_labels(self):
        return self.labels

#    def get_target(self):
#        return [1 if tweet_class == 'T' else 0 for tweet_class in self.target]

def print_positives(kw):

    y = kw.get_target()
    n = len(y)
    pos_ind = [i for i in range(n) if y[i]==1]
    n_pos = len(pos_ind)
    print n_pos*100.0 / n, n_pos, n


def run_classifier(kw):

    x = np.array

    n_trials = 10
    n_cv = 5
    print x.shape, y.shape
    clf = sklearn.naive_bayes.MultinomialNB()
    scores = np.zeros((n_trials * n_cv))
    for n in range(n_trials):
        x_shuffled, y_shuffled = sklearn.utils.shuffle(x, y, random_state=n)
        skf = cross_validation.StratifiedKFold(y_shuffled, n_folds=n_cv)  # random_state=n, shuffle=True)
        scores[n * n_cv:(n + 1) * n_cv] = cross_validation.cross_val_score(clf, x_shuffled, y_shuffled, cv=skf,
                                                                           scoring='roc_auc',
                                                                           verbose=2, n_jobs=1)
    return scores


def run_clusterer(dataset):

    hasher = HashingVectorizer(n_features=10000, stop_words='english', non_negative=True, norm=None, binary=False)
    #clusterer = DBSCAN(eps=0.5, min_samples=5, metric='cosine', algorithm='auto', leaf_size=30, p=None,
    #                             random_state=None)

    clusterer = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=1)

    X = TfidfVectorizer().fit_transform(dataset)

#    clf = make_pipeline(TfidfVectorizer(), clusterer)

    #clusterer.fit(X)

    #D = cosine_similarity(X)

    #print D.shape

    t = TSNE(n_components=2, init='pca', metric='euclidean')
    model = t.fit_transform(X)
    return model

def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Filename')

    arguments = parser.parse_args()

    filename = arguments.filename
    #filename="mandera_annotated.csv"
    kw = KenyanTweets(filename)
    #print_positives(kw)

    X = [tweet for tweet in kw]
    X_2D = run_clusterer(X)

    with open("output.txt", 'w') as fout:
        for row, label in zip(X_2D, kw.get_labels()):
            print row[0], row[1], label



if __name__ == "__main__":
    __main__()