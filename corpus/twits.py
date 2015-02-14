__author__ = 'verasazonova'

import csv
import numpy as np
import argparse
import re
import sklearn.naive_bayes
import sklearn.utils
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from utils import tsne
#from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer
from ttp import ttp
from medical import stop
from utils.tsne import tsne
import geograpy
import matplotlib.pyplot as plt
from models.transformers import D2VModel, LDAModel

def read_file_annotated(filename, x_ind=2, y_ind=3):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        y = []
        for row in reader:
            x.append(unicode(row[x_ind], errors='replace'))
            y.append(unicode(row[y_ind], errors='replace'))
    return x, y


def read_file(filename, x_ind=2, label_ind=16):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        x = []
        labels = []
        for row in reader:
            str = unicode(row[x_ind], errors='replace')
            x.append(str)
            str = unicode(row[label_ind], errors='replace')
            labels.append(str.replace(',', ', ').replace('/', '/ ').replace('-', '- '))
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
    def __init__(self, filename, annotated=True, x_ind=1, lbl_indx=15, include_RT=False):
        self.annotated = annotated
        self.include_RT = include_RT
        if self.annotated:
            self.data, self.labels = read_file_annotated(filename)
        else:
            self.data, self.labels = read_file(filename, x_ind=x_ind, label_ind=lbl_indx)

    def __iter__(self):
        p = MyParser()
        stop2 = stop+["makaburi", "rt"]
        for text in self.data:
            result = p.parse(text)
            parsed = re.sub(r'([,.?!])(?![\s.,!?])', r'\1 ', result.html)
            processed = []
            for word in parsed.split(' '):
                if word.lower() not in stop2:
                    processed.append(word.lower())
            slimmed = " ".join(processed)
            if self.include_RT:
                yield slimmed
            elif not parsed.startswith("RT"):
                yield slimmed

    def get_labels(self):
        if self.annotated:
            return [1 if label=='T' else 0 for label in self.labels]
        else:
            return self.labels

#    def get_target(self):
#        return [1 if tweet_class == 'T' else 0 for tweet_class in self.target]

def print_positives(kw):

    y = kw.get_target()
    n = len(y)
    pos_ind = [i for i in range(n) if y[i]==1]
    n_pos = len(pos_ind)
    print n_pos*100.0 / n, n_pos, n

'''
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
'''


def run_clusterer(dataset):

    #hasher = HashingVectorizer(n_features=10000, stop_words='english', non_negative=True, norm=None, binary=False)
    #clusterer = DBSCAN(eps=0.5, min_samples=5, metric='cosine', algorithm='auto', leaf_size=30, p=None,
    #                             random_state=None)

    #clusterer = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1, verbose=1)

    X = TfidfVectorizer(min_df=2, max_df=1.0, ngram_range=(1,3), use_idf=True, smooth_idf=False, norm=u'l2',
                        sublinear_tf=False, max_features=10000).fit_transform(dataset)



    corpus = [text.split() for text in dataset]

    lda = LDAModel(topn=15, no_below=2, no_above=1, mallet=False).fit_transform(corpus)

    print lda.shape

    #D = cosine_distances(X)
    #print "distances computed"
    #print D.shape

    #t = TSNE(n_components=2, init='random', metric='precomputed', perplexity=50)

    #data_2d = t.fit_transform(D)

    #model2 = tsne.tsne(X, coords=True)

    #return data_2d, lda
    return None, None


def get_locations(y):
    text_places = [place.title() for place in y]

    places = []
    cN=0
    cO = 0
    cA = 0
    cAm = 0
    cE = 0
    cK = 0
    for text in text_places:
        location = "None"
        if text:
            if re.search(r'Nairobi', text):
                location = "Nairobi"
                cN +=1
            elif re.search(r'Kenya', text):
                location = "Kenya"
                cK += 1
            elif re.search(r'Tanzania|Uganda|Africa|Liberia|Nigeria', text):
                location = "Africa"
                cA += 1
            elif re.search(r'Canada|Usa|Toronto|Vancouver|Montreal|San Francisco|New York|Chicago|Washington|Seattle|California|Dallas|Boston', text):
                location = "Americas"
                cAm += 1
            elif re.search(r'Europe|Uk|France|Germany|London|England|Croatia|Ukraine|Russia|The Netherlands|Manchester|Paris|Geneva', text):
                location = "Europe"
                cE += 1
            else:
                location = "Other"
                cO += 1
            #place = geograpy.get_place_context(text=text)
            #print place.cities
        places.append(location)
    #print y
    #print set(places)
    print cN, cK, cA, cAm, cE, cO

    return places

def run(arguments):
    filename = arguments.filename
    kw = KenyanTweets(filename, annotated=arguments.annotated, x_ind=int(arguments.xind), lbl_indx=int(arguments.lblind),
                      include_RT=arguments.rt)
    #print_positives(kw)

    y =kw.get_labels()
    if not arguments.annotated:
        places = get_locations(y)
    else:
        places = y
    lbs, indx = np.unique(places, return_inverse=True)

    #print lbs
    #print indx

    X = [tweet for tweet in kw]
    print len(X)
    print len(set(X))

    X_2D, lda = run_clusterer(X)

    #with open(filename+"-2d.txt", 'w') as fout:
    #    for row, label in zip(X_2D, indx):
    #        fout.write("%s, %s, %s\n" % (row[0], row[1], label))

    print "Data saved"
    return X_2D, indx, lda

def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Filename')
    parser.add_argument('-x', action='store', dest='xind', help='Filename')
    parser.add_argument('-l', action='store', dest='lblind', help='Filename')
    parser.add_argument('--rt', action='store_true', dest='rt', help='Filename')
    parser.add_argument('--an', action='store_true', dest='annotated', help='Filename')

    arguments = parser.parse_args()

    #filename="mandera_annotated.csv"

    #text_data = []
    #with open(arguments.filename) as f:
    #    for line in f:
    #        text_data.append ( map(float, line.strip().split(',')))

    #data = np.array(text_data)
    #print data.shape

    data, indx, lda = run(arguments)

    #lbls = [np.argmax(l) for l in lda]

    #print lbls

    #plt.scatter(data[:, 0], data[:, 1], c=lbls, linewidths=None, cmap=plt.cm.get_cmap('jet'))
    #plt.savefig(arguments.filename+".pdf")




if __name__ == "__main__":
    __main__()