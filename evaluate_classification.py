#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'Run this script for evaluate the object recognition score'

import matplotlib as mpl
mpl.use('Agg')

import argparse

import numpy as np
from numpy import genfromtxt
import seaborn as sb
import statistics

from analogymodel.custom_function import find_PCA, find_centroid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

from sklearn.neighbors import KNeighborsClassifier

# def plot_scores(stats):
    # plot scores
    # pyplot.hist(stats)
    # pyplot.show()

def confidence_interval(stats):
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    # ($lower + 0.5 * $diff, 0.5 * $diff);
    diff = upper - lower
    
    # print('%.2f confidence interval %.2f%% and %.2f%%' % (alpha*100, lower*100, upper*100))
    # print('%.2f confidence interval %.4f +- %.4f' % (alpha*100, lower + 0.5 * diff, 0.5 * diff))
    ret_str = '$%.4f\pm%.4f$' % (lower + 0.5 * diff, 0.5 * diff)
    return ret_str

def no_bootstrap(pred, label, output_mode):
    acc = accuracy_score(label, pred)
    pre = precision_score(label, pred, average='macro')
    rec = recall_score(label, pred, average='macro')
    f1 = 2 * (pre * rec) / (pre + rec)
    acc_str = '%.4f' % (acc)
    pre_str = '%.4f' % (pre)
    rec_str = '%.4f' % (acc)
    f1_str = '%.4f' % (f1)
    # format for latex table
    if output_mode == 1:
        print(acc_str)
    elif output_mode == 0:
        print(pre_str +  ' & ' + rec_str + ' & ' + f1_str)
    else:
        print(acc_str + ' & ' + pre_str +  ' & ' + rec_str + ' & ' + f1_str)

def bootstrap(pred, label, output_mode):
    n_iterations = 1000
    n_size = int(len(pred) * 0.50)
    # run bootstrap
    acc_arr = list()
    pre_arr = list()
    rec_arr = list()
    f1_arr = list()
    for i in range(n_iterations):
        # prepare train and test sets
        pred_s, label_s = resample(pred, label, n_samples=n_size)
        acc = accuracy_score(label_s, pred_s)
        pre = precision_score(label_s, pred_s, average='macro')
        rec = recall_score(label_s, pred_s, average='macro')
        f1 = 2 * (pre * rec) / (pre + rec)
        acc_arr.append(acc)
        pre_arr.append(pre)
        rec_arr.append(rec)
        f1_arr.append(f1)
    # confidence intervals
    acc_str = confidence_interval(acc_arr)
    pre_str = confidence_interval(pre_arr)
    rec_str = confidence_interval(rec_arr)
    f1_str = confidence_interval(f1_arr)
    # format for latex table
    if output_mode == 1:
        print(acc_str)
    elif output_mode == 0:
        print(pre_str +  ' & ' + rec_str + ' & ' + f1_str)
    else:
        print(acc_str + ' & ' + pre_str +  ' & ' + rec_str + ' & ' + f1_str)

def centroid_knn(data, ref_data, label, ref_label, using_boostrap=False, output_mode=0):
    # clf = KNeighborsClassifier(n_neighbors=5)
    clf = NearestCentroid()
    clf.fit(ref_data, ref_label)
    pred = clf.predict(data)
    # print(confusion_matrix(pred, label))
    # print(classification_report(pred, label, digits=4))
    if using_boostrap:
        bootstrap(pred, label, output_mode)
    else:
        no_bootstrap(pred, label, output_mode)

def mlp(data, ref_data, label, ref_label, lr=.1, iter=20, using_boostrap=False, output_mode=0):
    mlp = MLPClassifier(hidden_layer_sizes=(800,800), max_iter=iter,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=lr)
    mlp.fit(ref_data, ref_label)
    pred = mlp.predict(data)
    # print(confusion_matrix(pred, label))
    # print(classification_report(pred, label, digits=4))
    if using_boostrap:
        bootstrap(pred, label, output_mode)
    else:
        no_bootstrap(pred, label, output_mode)

def knn(data, ref_data, label, ref_label, nn=5, using_boostrap=False, output_mode=0):
    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(ref_data, ref_label)
    pred = neigh.predict(data)
    if using_boostrap:
        bootstrap(pred, label, output_mode)
    else:
        no_bootstrap(pred, label, output_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the object recognition score. \
        By using the embedding of each class as the references and evaluate to the closet one.')
    parser.add_argument('--ref-data', type=str, default='mnist-train-embedding.txt',
                    help='set references data file (default:mnist-train-embedding.txt)')
    parser.add_argument('--ref-label', type=str, default='mnist-train-label.txt',
                    help='set references label file (default:mnist-train-label.txt)')
    parser.add_argument('--data', type=str, default='mnist-test-embedding.txt',
                    help='set data file (default:mnist-test-embedding.txt)')
    parser.add_argument('--label', type=str, default='mnist-test-label.txt',
                    help='set label file (default:mnist-test-label.txt)')
    parser.add_argument('--mode', type=str, default='cknn',
                    help='choose mode knn or mlp (default:cknn)')
    parser.add_argument('--lr', type=float, default=.1,
                    help='set number of iterations (default:.1)')
    parser.add_argument('--iter', type=int, default=20,
                    help='set number of iterations (default:20)')
    parser.add_argument('--nn', type=int, default=5,
                    help='set k for knn (default:5)')
    parser.add_argument('--use-cosine', action='store_true', default=False,
                    help='using cosine (default: false, using euclidean distance)')
    parser.add_argument('--use-boostrap', action='store_true', default=False,
                    help='using boostrap confidence value (default: false)')
    parser.add_argument('--output-mode', type=int, default=0,
                    help='set the output mode 0: Precision, Recall, F-1 1: Accurarcy 2: Both (default: 0)')
    parser.add_argument('--input', type=str, default=None,
                    help='load file')
    args = parser.parse_args()

    if args.input != None:
        args.ref_data = args.input + '-train-embedding.txt' 
        args.data = args.input + '-test-embedding.txt' 
        args.ref_label = args.input + '-train-label.txt' 
        args.label = args.input + '-test-label.txt' 

    ref_data = genfromtxt(args.ref_data, delimiter=' ')
    ref_label = genfromtxt(args.ref_label, delimiter=' ')
    data = genfromtxt(args.data, delimiter=' ')
    label = genfromtxt(args.label, delimiter=' ')
    
    if args.mode == 'mlp':
        mlp(data, ref_data, label, ref_label, lr=args.lr, iter=args.iter, using_boostrap=args.use_boostrap, output_mode=args.output_mode)
    elif args.mode == 'knn':
        knn(data, ref_data, label, ref_label, nn=args.nn, using_boostrap=args.use_boostrap, output_mode=args.output_mode)
    else:
        centroid_knn(data, ref_data, label, ref_label, using_boostrap=args.use_boostrap, output_mode=args.output_mode)