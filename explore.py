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
from operator import itemgetter

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

def visualize(arguments):

    filename = get_filename(arguments.dataset)
    corpus = MedicalReviewAbstracts(filename, ['T', 'A', 'M'], labeled=False, tokenize=False)
    x = [text for text in corpus]
    y = corpus.get_target()


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


def classify_by_sim(arguments):

    threshs = [0.5, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9]

    for dataset in arguments.dataset:

        print dataset

        accs = []
        tups = []

        filename = get_filename(dataset)
        corpus = MedicalReviewAbstracts(filename, ['T', 'A'], labeled=True)

        data = [text.labels[0] for text in corpus]
        y = corpus.get_target()
        n = len(y)

        maj = len([ c for c in y if c == -1 ]) / float(len(y))


        d2v_model = Doc2Vec(corpus, size=300, alpha=0.025, window=10, min_count=2, sample=0, seed=1,
                                            workers=4, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0,
                                            train_words=True, train_lbls=True, initial=arguments.modelname)


        for thres in threshs:

            correct_pos = 0.0
            false_pos = 0.0
            correct_neg = 0.0
            false_neg = 0.0
            uncls = 0.0
            total_model = 0.0
            total = len(data)

            n_neighb = 2

            cls_dict = {}

            for name, cls in zip(data, y):
                cls_dict[name] = cls


            for i in range(n):
                train_x = data[0:i-1]+data[i+1:]
                test_x = data[i]
                train_y = y[0:i]+y[i+1:]
                test_y = y[i]

                if test_x in d2v_model:

                    # most similar texts (not words) with similarity more than 0.2
                    most_sim_texts = [(name, sim) for (name, sim) in d2v_model.most_similar_cosmul([test_x], topn=300)
                                      if name in train_x and sim > thres]

                    if most_sim_texts:
                        classes = [cls_dict[name]*sim for (name, sim) in most_sim_texts[0:min(len(most_sim_texts), n_neighb)]]
                        if np.array(classes).mean() > 0:
                            cls = 1
                        else:
                            cls = -1
                    else:
                        classes = []
                        cls = None

                    #if cls is not None:
                    #    print cls, test_y, classes

                    if cls is None:
                        uncls += 1
                    elif cls == 1:
                        if cls == test_y:
                            correct_pos += 1
                        else:
                            false_pos += 1
                    elif cls == -1:
                        if cls == test_y:
                            correct_neg += 1
                        else:
                            false_neg += 1

                    total_model += 1

            #print correct_pos, correct_neg
            #print false_pos, false_neg
            t = (correct_pos+correct_neg+false_neg+false_pos)
            if t > 0:
                acc = (correct_pos+correct_neg)/t
            else:
                acc = 0
            #maj = (correct_neg+false_neg)/ (correct_pos+correct_neg+false_neg+false_pos)
            N = total_model - uncls
            print thres, acc, maj, N, correct_pos, correct_neg, false_pos, false_neg
            accs.append(acc)
            tups.append((correct_pos, correct_neg, false_pos, false_neg))

        plt.clf()
        plt.plot(threshs, accs, '-b', threshs, accs, '.g', [threshs[0], threshs[len(threshs)-1]], [maj, maj], '-k')
        plt.ylim([accs[0], 1.01])
        plt.savefig(dataset+".pdf")


def check_similarity_class(arguments):
    filename = get_filename(arguments.dataset)
    corpus =  MedicalReviewAbstracts(filename, ['T', 'A'], labeled=True)
    class_label = corpus.get_target()


    d2v_model = Doc2Vec(corpus, size=100, alpha=0.025, window=5, min_count=5, sample=0, seed=1,
                                        workers=4, min_alpha=0.0001, dm=1, hs=1, negative=0, dm_mean=0,
                                        train_words=False, train_lbls=True, initial=arguments.modelname)

    print "d2v model created"

    class_label_dict = {}

    for text, cls in zip(corpus, class_label):
        class_label_dict[text.labels[0]] = cls

    #d2v_model_name = arguments.modelname
    #d2v_model = Doc2Vec.load_word2vec_format(d2v_model_name, binary=True)

    p_same = []
    p_diff = []
    n_total = 0
    n_notinmodel = 0
    n_nosims = 0
    n_inmodel = 0

    for text in corpus:
        n_same = 0
        n_diff = 0
        if text.labels[0] in d2v_model:
            most_sim_texts = [tup for tup in d2v_model.most_similar_cosmul([text.labels[0]], topn=100)
                              if tup[0] in class_label_dict and tup[1] > 0.2]

            if not most_sim_texts:
                most_sim_text = []
            else:
                n_neighb = min(len(most_sim_texts), 5)
                most_sim_text = most_sim_texts[0:n_neighb]
                n_nosims += 1

            for lbl, sim in most_sim_text:
                print lbl, sim
                if sim > 0.2:
                    if class_label_dict[lbl] == class_label_dict[text.labels[0]]:
                        n_same += 1
                    else:
                        n_diff += 1
                else:
                    n_nosims += 1

            n_inmodel += 1
            if (n_same + n_diff) > 0:
                p_same.append(float(n_same) / (n_same + n_diff))
                p_diff.append(float(n_diff) / (n_same + n_diff))
        else:
            n_notinmodel += 1

        n_total += 1

    p_same = np.array(p_same)
    p_diff = np.array(p_diff)
    print p_same.mean(), p_diff.mean()
    total = 0
    for x, y in zip(p_same, p_diff):
        if x > y:
            total += 1

    print total / float(len(p_same))
#    plt.plot(p_same, '.k')
    #    plt.show()

def cluster_words(arguments):

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
    parser.add_argument('-d', action='store', nargs='+', dest='dataset', help='Dataset')
    parser.add_argument('--model', action='store', dest='modelname', help='W2V model name')
    parser.add_argument('--corpus', action='store', dest='corpusname', help='D2V corpus name')

    arguments = parser.parse_args()

    #visualize(arguments)
#    check_similarity_class(arguments)
    classify_by_sim(arguments)

if __name__ == "__main__":
    __main__()