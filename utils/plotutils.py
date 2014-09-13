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

def plot_base(x, y, color='b', s=50):
    plt.scatter(x, np.mean(y), s=s, marker="*", color=color)

def plot_xy(x_str, scores, x_label, y_label, filename, color='b', s=50, score_base=None):
    cpool = ['white', color]
    cmap3 = col.ListedColormap(cpool, 'indexed')
    cm.register_cmap(cmap=cmap3)

    x, x_tags = to_number(x_str)

    y = map(np.mean, scores)

    print x, y
    if score_base is None:
        score_base = scores[0]
    colors = [1 if stat_different(score, score_base) else 0 for score in scores]

    plt.plot(x, y, ls='--', c=color, label=filename)
    plt.scatter(x, y, s=s, cmap=cmap3, marker='o', c=colors, vmin=0, vmax=1,
                linewidths=(1,), alpha=1, edgecolors=color)
    if x_tags is not None:
        plt.xticks(x, x_tags, rotation=6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def to_number(a):
    try:
        data = np.array(a, dtype=float)
        return data, None
    except ValueError:
        u, ind = np.unique(a, return_index=True)
        return ind, u[ind]


def readscores(filename):
    scores = []
    x = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            x.append(values[0])
            scores.append(np.array(map(float, values[1:])))

    return x, scores


def __main__():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', action='store', nargs="+", dest='filename', help='Data filename')
    parser.add_argument('-l', action='store', nargs="+", dest='labels', help='Data filename')
    parser.add_argument('-x', action='store', dest='xlabel', default="", help='Data filename')
    parser.add_argument('-y', action='store', dest='ylabel', default="", help='Data filename')
    parser.add_argument('-o', action='store', dest='output', default="output", help='Data filename')
    parser.add_argument('-t', action='store', dest='title', default="", help='Data filename')
    parser.add_argument('-n', action='store', dest='base_ind', default="0", help='Data filename')

    arguments = parser.parse_args()

    '''
    if arguments.filename is None:
        filename = "/home/vera/Work/TextVisualization/Experiments/40/units_Antihistamines.txt_ldalog.txt"
    else:
        filename = arguments.filename
    '''

    colors = ['g', 'r', 'b', 'k', 'y']
    score_base = None
    base_ind = int(arguments.base_ind)
    for i, filename in enumerate(arguments.filename):
        x, scores = readscores(filename)
        if i==base_ind:
            # calculate statistical significance with respect to the very first datapoint
            score_base = scores[0]
        plot_xy(x, scores, filename, "roc", "", color=colors[i], s=80, score_base=score_base)
        if i==base_ind:
            plot_base(float(x[0]), score_base, color=colors[0])


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

    if arguments.labels is None:
        labels = arguments.filename
    else:
        labels = arguments.labels

    plt.legend(labels, loc=0)
    plt.xlabel(arguments.xlabel)
    plt.ylabel(arguments.ylabel)
    plt.grid()
    plt.title(arguments.title)
    plt.savefig(arguments.output + ".pdf")


if __name__ == "__main__":
    __main__()