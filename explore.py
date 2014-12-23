__author__ = 'verasazonova'


from gensim.models import Word2Vec, Doc2Vec
from models.transformers import D2VModel, CRPClusterer, W2VList
import numpy as np
import argparse
from utils.tsne import tsne
from sklearn.pipeline import Pipeline
from corpus.medical import get_filename, MedicalReviewAbstracts
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

def cosine_product(x):
    N = x.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        D[i, i] = 0
        len_i = np.sqrt(x[i].dot(x[i]))
        for j in range(i, N):
            len_j = np.sqrt(x[j].dot(x[j]))
            if len_i == 0 or len_j == 0:
                D[i, j] = 0
            else:
                D[i, j] = np.dot(x[i], x[j]) / (len_i * len_j)
                D[j, i] = D[i, j]
    return D

def visualize(x, y):


    stop = set([word.strip() for word in open("/Users/verasazonova/Work/PycharmProjects/textsim/corpus/stopwords.txt",
                                              'r').readlines()])

    vectorizer = TfidfVectorizer(min_df=2, max_df=1.0, ngram_range=(1,3), stop_words=stop,
                                 use_idf=True, smooth_idf=False, norm=u'l2', sublinear_tf=False, max_features=10000)
    #input=u'content', encoding=u'utf-8', charset=None, decode_error=u'replace',
    #                             charset_error=None, strip_accents=None, lowercase=True, preprocessor=None,
    #                             tokenizer=None, analyzer=u'word', stop_words=None, token_pattern=u'(?u)\b\w\w+\b',
    #                             ngram_range=(1, 1), max_df=0.9, min_df=2, max_features=None, vocabulary=None,
    #                             binary=False, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

#    print x[0]
    bow = vectorizer.fit_transform(x)

    #print bow.toarray().shape
    similarities = cosine_product(bow.toarray()) #
    #similarities = cosine_similarity(bow.toarray())
    print similarities.shape

    # enforce self, self 0 probability
    for i in range(bow.shape[0]):
        similarities[i, i] = 0
    #distances = 1 - cosine_similarity(bow)

    #t = TSNE(n_components=2, init='random', metric='precomputed')
    #data_2d = t.fit_transform(distances)

    data_2d = tsne( similarities, coords=False, perform_pca=False, perplexity=50)


    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y)
    plt.savefig("output.pdf")


def explore(arguments):

    filename = get_filename(arguments.dataset)
    corpus = [text for text in MedicalReviewAbstracts(filename, ['T', 'A'], labeled=False)]


    w2v_model_name = arguments.modelname
    w2v_model = Word2Vec.load_word2vec_format(w2v_model_name, binary=True)
    w2v_model.init_sims(replace=True)

    print
    print " ".join(corpus[0])

    model = Pipeline ([
            ('w2v', W2VList(w2v_model=w2v_model)),
            ('crp', CRPClusterer()) ])

    ids = model.fit_transform([corpus[0]])

#    print ids[0]
    print
    for cluster in ids[0]:
        print " ".join(set( [corpus[0][id] for id in cluster]))
        print "\n"

    titles = [text for text in MedicalReviewAbstracts(filename, ['T'], labeled=False)]

    print " ".join(titles[0])

#    d2v_corpus = MedicalReviewAbstracts(arguments.corpusname, ['T', 'A'], labeled=True)
#    d2v_model = Doc2Vec(d2v_corpus, train_words=True, train_lbls=True, initial=w2v_model_name,
#                        size=w2v_model.layer1_size, window=5)
#    d2v_model.save_word2vec_format("d2v_100_5")

    d2v_model = Doc2Vec.load_word2vec_format("d2v_100_5")

    corpus_lbld = [text for text in MedicalReviewAbstracts(filename, ['T', 'A'], labeled=True)]


    print d2v_model.most_similar(positive=[corpus_lbld[0].labels[0]])

    print d2v_model.most_similar(positive=[corpus_lbld[1].labels[0]])


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', action='store', dest='dataset', help='Dataset')
    parser.add_argument('--model', action='store', dest='modelname', help='W2V model name')
    parser.add_argument('--corpus', action='store', dest='corpusname', help='D2V corpus name')

    arguments = parser.parse_args()

    filename = get_filename(arguments.dataset)
    corpus = MedicalReviewAbstracts(filename, ['T', 'A', 'M'], labeled=False, tokenize=False)
    x = [text for text in corpus]
    y = corpus.get_target()

    visualize(x, y)

if __name__ == "__main__":
    __main__()