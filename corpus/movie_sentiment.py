__author__ = 'verasazonova'

import os.path
import argparse
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
import numpy as np
from sklearn import linear_model, metrics, grid_search, svm
from models.transformers import D2VModel, BOWModel, ReadVectorsModel
from sklearn.pipeline import Pipeline, FeatureUnion

def normalize(phrase, min_count):
    norm_phrase = phrase.lower()
    for punctuation in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/', '\"', '\'', '^']:
        norm_phrase = norm_phrase.replace(punctuation, punctuation+' ')
    data = norm_phrase.split(' ')
    data_length = len(data)
    if data_length < min_count:
        data = (['null'] * (min_count - data_length)) + data
    return data


class IMDBCorpus:
    def __init__(self, filename, min_count):
        self.filename = filename
        self.min_count =  min_count

    def __iter__(self):
        with open(self.filename, 'r') as f:
            cnt = 0
            for line in f:
                yield LabeledSentence(normalize(line, self.min_count), ["_*" + str(cnt)])
                cnt += 1

    def get_train(self):
        with open(self.filename, 'r') as f:
            cnt = 0
            for line in f:
                if cnt < 25000:
                    yield LabeledSentence(normalize(line, self.min_count), ["_*" + str(cnt)])
                cnt += 1


    def get_test(self):
        with open(self.filename, 'r') as f:
            cnt = 0
            for line in f:
                if cnt >= 25000 and cnt < 50000:
                    yield LabeledSentence(normalize(line, self.min_count), ["_*" + str(cnt)])
                cnt += 1

    def get_target_train(self):
        data = np.ones((25000))
        data[12500:] = -1*data[12500:]
        return data


    def get_target_test(self):
        return self.get_target_train()


def find_split(phrase, sentences):
    for sentence, split in sentences.itervalues():
        if sentence.find(phrase) >= 0:
            return split
        elif sentence.replace('-LRB-', '(').replace('-RRB-', ')').lower().find(phrase.lower()) >= 0:
            return split
    #print phrase
    return None


def read_phrases(dirname):
    dictfilename = "dictionary.txt"
    sentfilename = "sentiment_labels.txt"
    sentencefilename = "datasetSentences.txt"
    splitfilename = "datasetSplit.txt"

    phrases = {}
    with open(os.path.join(dirname, dictfilename), 'r') as f:
        for line in f:
            data = line.split('|')
            id = int(data[1])
            if id not in phrases:
                phrases[id] = data[0]

    print "Phrases read"
    sentiments = {}
    with open(os.path.join(dirname, sentfilename), 'r') as f:
        f.readline()
        for line in f:
            data = line.split('|')
            id = int(data[0])
            if id not in sentiments:
                sentiments[id] = float(data[1].strip())

    print "Sentiments read"

    sentences = {}
    with open(os.path.join(dirname, sentencefilename), 'r') as f:
        f.readline()
        for line in f:
            data = line.split('\t')
            id = int(data[0])
            phrase = data[1]
            if id not in sentences:
                sentences[id] = phrase
    print "Sentences read"

    with open(os.path.join(dirname, splitfilename), 'r') as f:
        f.readline()
        for line in f:
            data = line.split(',')
            id = int(data[0])
            split = int(data[1])
            if id in sentences:
                sentences[id] = (sentences[id], split)

    print "Splits read"

    phrases_splits = {}
    with open(os.path.join(dirname, "phrases_split.txt"), 'r') as f:
        f.readline()
        for line in f:
            data = line.split('|')
            id = int(data[0].strip())
            if data[1].strip() == "None":
                split = 0
            else:
                split = int(data[1].strip())
            if id not in phrases_splits:
                phrases_splits[id] = split

    print "Phrase splits read"


    # with open(os.path.join(dirname, "phrases_split.txt"), 'w') as fout:
    #     fout.write("phrase_id|split\n")
    #     for id in phrases.iterkeys():
    #         split = find_split(phrases[id], sentences)
    #         phrases_splits[id] = split
    #         fout.write("%s|%s\n" % (id, split))
    # print "Splits to phrases assigned"

    return phrases, sentiments, phrases_splits


class MovieSentimentsCorpus:
    def __init__(self, dirname, binary=True):
        self.phrases, self.sentiments, self.phrases_split = read_phrases(dirname)
        self.binary = binary


    def normalize(self, id):
        norm_phrase = self.phrases[id].lower()
        for punctuation in [',', ':', '.', '(', ')', '!', '?', ':', ';', '/']:
            norm_phrase = norm_phrase.replace(punctuation, punctuation+' ')
        return LabeledSentence(norm_phrase.split(' '), [str(id)])


    def __iter__(self):
        for id in sorted(self.phrases.iterkeys()):
            yield self.normalize(id)

    def get_train(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 1:
                yield self.normalize(id)

    def get_test(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 2:
                yield self.normalize(id)

    def get_dev(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 3:
                yield self.normalize(id)


    def coarse_grain(self, sentiment):
        if self.binary:
            if sentiment >=0 and sentiment <= 0.5:
                sent_coarse = 0
            elif sentiment <=1:
                sent_coarse = 1
            else:
                sent_coarse = None
                print "Error unknown sentiment"
        else:
            if sentiment >= 0 and sentiment <= 0.2:
                sent_coarse = 0
            elif sentiment <= 0.4:
                sent_coarse = 1
            elif sentiment <= 0.6:
                sent_coarse = 2
            elif sentiment <= 0.8:
                sent_coarse = 3
            elif sentiment <= 1:
                sent_coarse = 4
            else:
                sent_coarse = None
                print "Error unknown sentiment"
        return sent_coarse

    def get_target(self):
        for id in sorted(self.phrases.iterkeys()):
            yield self.coarse_grain( self.sentiments[id] )

    def get_target_train(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 1:
                yield self.coarse_grain( self.sentiments[id] )


    def get_target_test(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 2:
                yield self.coarse_grain( self.sentiments[id] )


    def get_target_dev(self):
        for id in sorted(self.phrases.iterkeys()):
            if self.phrases_split[id] == 3:
                yield self.coarse_grain( self.sentiments[id] )


def d2v_to_array(corpus, d2v_model, size):
    docs = [text  for text in corpus]
    n_docs = len(docs)
    x_train = np.zeros((n_docs, size))
    for doc_cnt, text in enumerate(docs):
        if text.labels[0] in d2v_model:
            x_train[doc_cnt, 0:size] = d2v_model[text.labels[0]]

    return x_train


def run_d2v(corpus):
    size = 100
    alpha = 0.025
    min_alpha = 0.001
    window = 10
    min_count = 1
    num_iters = 20

    d2v_model = Doc2Vec(sentences=None, size=size, alpha=alpha, window=window,
                                            min_count=min_count, seed=1,
                                            workers=4, min_alpha=min_alpha, dm=1, hs=0,
                                            dm_mean=0, negative=5, sample=0.0001,
                                            train_words=False, train_lbls=False)

    d2v_model.build_vocab(corpus)

    # train model on train

    d2v_model.train_lbls = True
    d2v_model.train_words = True

    if num_iters == 1:
        d2v_model.train(corpus.get_train())
    else:
        for i in range(0, num_iters+1):
            #d2v_model.alpha -= 0.002
            #d2v_model.min_alpha = d2v_model.alpha
            d2v_model.train(corpus.get_train())

    print d2v_model

    x_train = d2v_to_array(corpus.get_train(), d2v_model, size)
    print x_train.shape

    y_train = np.array([y for y in corpus.get_target_train()])
    print y_train.shape

    d2v_model.train_lbls = True
    d2v_model.train_words = False

    if num_iters == 1:
        d2v_model.train(corpus.get_train())
    else:
        for i in range(0, num_iters+1):
            #d2v_model.alpha -= 0.002
            #d2v_model.min_alpha = d2v_model.alpha
            d2v_model.train(corpus.get_train())

    x_test = d2v_to_array(corpus.get_test(), d2v_model, size)
    y_test =  np.array([ y for y in corpus.get_target_test()])

    print x_test.shape, y_test.shape

    clf = linear_model.LogisticRegression()
    clf.C = 10000
    #clf.fit(x_train, y_train)

    parameters  = {'C': [1000, 6000, 10000]}




    #y_predicted = clf.predict(x_test)
    #scores =  metrics.precision_score(y_test, y_predicted)
    #print scores
#    print scores.mean(), scores.std()

    #d2v_model.save_word2vec_format("sent_model.bin", binary=True)
def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store', dest='directory', help='Directory')
    parser.add_argument('-v', action='store', dest='vectors', help='Directory')

    arguments = parser.parse_args()

    #corpus = MovieSentimentsCorpus(arguments.directory)
    corpus = IMDBCorpus(arguments.directory, 9)

    #data = np.loadtxt(arguments.vectors)
    #labels = data[:, 0]
    #vectors = data[:, 1:]


    clf = linear_model.LogisticRegression(C=100)

    #clf = svm.SVC(C=10)

    clf_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('d2v', D2VModel(d2v_model=None, corpus=corpus, alpha=0.05, size=400, window=10, min_count=1,
                             min_alpha=0.0001, num_iters=20, initial_w2v=None, two_models=True, negative=5,
                             sample=0.0001, hs=0, dm=0, seed=0, num_iters_words=None, alpha_words=None,
                             train_type="fixed")),
            #('d2v', ReadVectorsModel("sentence_vectors.txt")),
            #('bow', BOWModel(no_below=2, no_above=0.9))
                ])),
        ('clf', clf) ])

    parameters = {
#                   'features__d2v__num_iters': [10, 20],
                   'features__d2v__alpha': [0.025, 0.05, 0.1]
#                   'features__d2v__train_type': ["fixed", "decreasing"]
                   }


    grid_clf = grid_search.GridSearchCV(clf_pipeline, param_grid=parameters, iid=False, cv=3, refit=True)


    #x_dev = np.array([text for text in corpus.get_dev()])
    #y_dev =  np.array([ y for y in corpus.get_target_dev()])


    x_dev = np.array([text for text in corpus.get_train()])
    y_dev =  np.array([ y for y in corpus.get_target_train()])

    print x_dev.shape, y_dev.shape

    #grid_clf.fit(x_dev, y_dev)

#    print grid_clf.grid_scores_
##    print grid_clf.best_params_
#    print grid_clf.best_score_

#    best_clf = grid_clf.best_estimator_
    best_clf = clf_pipeline

    x_train = np.array([text for text in corpus.get_train()])
    y_train =  np.array([ y for y in corpus.get_target_train()])

    x_test = np.array([text for text in corpus.get_test()])
    y_test =  np.array([ y for y in corpus.get_target_test()])


    best_clf.fit(x_train, y_train)
    print best_clf
    print x_test.shape, y_test.shape
    y_predicted = best_clf.predict(x_test)

    print metrics.precision_score(y_test, y_predicted)


if __name__ == "__main__":
    __main__()