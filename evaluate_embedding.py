#!/usr/bin/env python3

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '30/06/2020', '1.0'
__description__ = 'Run this script for create the embedding result'

import argparse

import matplotlib
matplotlib.use('Agg') # For save the figure without showing
import numpy as np

# For plot 3D graph
from mpl_toolkits.mplot3d import Axes3D 
from numpy import genfromtxt

import statistics

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from analogymodel.custom_function import find_PCA, find_centroid
from analogymodel.custom_ploter import *

from tsne import bh_sne
# tsne from https://pypi.org/project/tsne/
def plot_bh_tsne(name, data, label, k=10, is_show=False, limit_x=None, limit_y=None):
    X = np.asarray(data, dtype=np.float64)
    data_bh_tsne = bh_sne(X)
    plot_embedding_2d(name, data_bh_tsne, label, k, is_show, limit_x=limit_x, limit_y=limit_y)

from sklearn.manifold import TSNE
def plot_tsne(name, data, label, k=10, is_show=False, limit_x=None, limit_y=None):
    X = np.asarray(data, dtype=np.float64)
    RS = 123
    data_tsne = TSNE(random_state=RS).fit_transform(X)
    plot_embedding_2d(name, data_tsne, label, k, is_show, limit_x=limit_x, limit_y=limit_y)

from sklearn.decomposition import PCA
def plot_pca_skl(name, data, label, k=10, is_show=False, limit_x=None, limit_y=None):
    X = np.asarray(data, dtype=np.float64)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(X)
    plot_embedding_2d(name, data_pca, label, k, is_show, limit_x=limit_x, limit_y=limit_y)


def plot_pca_50(name, data, label, k=10, is_show=False, limit_x=None, limit_y=None):
    X = np.asarray(data, dtype=np.float64)
    pca = PCA(n_components=50)
    data_pca = pca.fit_transform(X)
    plot_embedding_2d(name, data_pca, label, k, is_show, limit_x=limit_x, limit_y=limit_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script provide several sub-scripts to analyze the embedding')
    parser.add_argument('--data', type=str, default='mnist-test-embedding.txt',
                    help='set data file (default:mnist-test-embedding.txt)')
    parser.add_argument('--label', type=str, default='mnist-test-label.txt',
                    help='set label file (default:mnist-test-label.txt)')
    parser.add_argument('--number-of-class', type=int, default=10,
                    help='set the number of class (default:10)')
    parser.add_argument('--limit-x', type=int, default=0,
                    help='set limit x axis value (default:0 (no limit))')
    parser.add_argument('--limit-y', type=int, default=0,
                    help='set limit y axis value (default:0 (no limit))')
    parser.add_argument('--output', type=str, default='analogy',
                    help='set the output file name (default:\'analogy\')')
    parser.add_argument('--input', type=str, default=None,
                    help='set the input file name')
    args = parser.parse_args()

    if args.input != None:
        args.data = args.input + '-embedding.txt' 
        args.label = args.input + '-label.txt' 

    data = genfromtxt(args.data, delimiter=' ')
    label = genfromtxt(args.label, delimiter=' ')

    # plot_emb_distance('hist', data, label, k=args.number_of_class)
    # plot_pca_2d('pca', data, label, k=args.number_of_class)
    # plot_cosine('mnist-hist-fc', data, label, 10)
    # plot_embedding('mnist', data, label, 2, 10, True)

    limit_x = None
    limit_y = None
    if args.limit_x != 0:
        limit_x = [-1*args.limit_x, args.limit_x]
    if args.limit_y != 0:
        limit_y = [-1*args.limit_y, args.limit_y]

    plot_pca_skl(args.output+'-PCA', data, label, k=args.number_of_class)
    plot_tsne(args.output+'-t-SNE', data, label, k=args.number_of_class, limit_x=limit_x, limit_y=limit_y)
    # plot_bh_tsne(args.custom_name+'-bh-t-SNE', data, label, k=args.number_of_class)