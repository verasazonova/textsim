__author__ = 'verasazonova'

import csv
import numpy as np
import re
import sklearn.naive_bayes
import sklearn.utils
from ttp import ttp

def read_file(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        y = []
        for row in reader:
            x.append(row[2].decode('utf-8'))
            y.append(row[3])
    return x, y


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
        self.data, self.target = read_file(filename)

    def __iter__(self):
        p = MyParser()
        for text in self.data:
            result = p.parse(text)
            yield re.sub(r'([,.?!])(?![\s.,!?])', r'\1 ', result.html).split()

    def get_target(self):
        return [1 if tweet_class == 'T' else 0 for tweet_class in self.target]

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

def __main__():
    filename="gikomba_annotated.csv"
    #filename="mandera_annotated.csv"
    kw = KenyanTweets(filename)
    print_positives(kw)


    #for tweet, tweet_class in zip(kw, kw.get_target()):
    #    if tweet_class == 1:
    #        print tweet


if __name__ == "__main__":
    __main__()