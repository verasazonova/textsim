# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 12:55:00 2014

@author: vera
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import matplotlib.colors as col
import matplotlib.cm as cm


def stat_different(x1, x2):
    thresh = 0.05
    t, p = stats.ttest_rel(x1, x2)
    if p < thresh:
        return True
    return False


def plot_xy(x, scores, x_label, y_label, filename, color='b', s=50):
    cpool = ['black', color]
    cmap3 = col.ListedColormap(cpool, 'indexed')
    cm.register_cmap(cmap=cmap3)

    y = map(np.mean, scores)
    score_base = scores[0]
    colors = [1 if stat_different(score, score_base) else 0 for score in scores]
    plt.plot(x, y, ls='--', c=color, label=filename)
    plt.scatter(x, y, s=s, cmap=cmap3, marker='o', c=colors,
                linewidths=(0,), alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    print colors


def readscores(filename):
    scores = []
    x = []
    with open(filename, 'r') as f:
        for line in f:
            x.append(line.split()[0])
            scores.append(np.array(map(float, line.split()[1:])))

    return x, scores


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', dest='filename', help='Data filename')
    arguments = parser.parse_args()

    if arguments.filename is None:
        filename = "/home/vera/Work/TextVisualization/Experiments/40/units_Antihistamines.txt_ldalog.txt"
    else:
        filename = arguments.filename

    x, scores = readscores(filename)
    plot_xy(x, scores, "n_topics", "roc", "lda", color='g', s=80)

    # filename = "/home/vera/Work/Spyder/textsim/textsim/medabs_mlda_log3.txt"
    #x, scores = readscores(filename)
    #plot_xy(x, scores, "n_topics", "roc", "mlda", color='g', s=100)

    """
    similarities = np.loadtxt("/home/vera/Work/TextVisualization/Experiments/40/Estrogens_tam_vector.txt")
    y1 = similarities.mean()
    plt.plot([0, 20], [y1, y1], 'b-', label="vector")

    similarities2 = np.loadtxt("/home/vera/Work/TextVisualization/Experiments/40/Estrogens_tam_path.txt")
    y2 = similarities2.mean()
    plt.plot([0, 20], [y2, y2], 'y-', label="path")
    """

    plt.legend(loc=2)
    plt.grid()
    plt.title(filename)
    plt.savefig(filename+".pdf")


if __name__ == "__main__":
    __main__()