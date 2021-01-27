#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'My custom plot functions'

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import torch

from analogymodel.custom_function import find_PCA, find_centroid


def plot_sample(name, imgs, labels, is_show=False):
    fig = plt.figure(figsize=(18, 8), dpi=80)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(imgs[i], cmap='gray', interpolation='none')
        plt.title(labels[i])
        plt.xticks([])
        plt.yticks([])
    plt.savefig('result/'+ name + '-samples.png')
    if is_show :
        plt.show()


def plot_graph(name, title, label, data, x_title=None, y_title=None, xlim=None, ylim=None, loc='upper right', grid=False, is_show=False):
    fig = plt.figure()
    if grid == True:
        plt.grid()
    for d in data:
        plt.plot(d[0], d[1])
    plt.title(title)
    plt.legend(label, loc=loc)
    if xlim :
        plt.xlim(xlim)
    if ylim :
        plt.ylim(ylim)
    if x_title is not None:
        plt.xlabel(x_title)
    if y_title is not None:
        plt.ylabel(y_title)
    plt.tight_layout()
    plt.savefig('result/'+ name +'-plot.png')
    if is_show :
        plt.show()


def plot_embedding(name, data, label, dim, k=10, using_PCA=False, is_show=False):
    if using_PCA :
        if dim == 2:
            plot_pca_2d(name, data, label, k, is_show)
        elif dim == 3:
            plot_pca_3d(name, data, label, k, is_show)
    else:
        if dim == 2:
            plot_embedding_2d(name, data, label, k, is_show)
        elif dim == 3:
            plot_embedding_3d(name, data, label, k, is_show)


def plot_embedding_2d(name, data, label, k=10, is_show=False, limit_x=None, limit_y=None):
    plt.figure()
    # plt.figure(figsize=(8,8))
    for i in range(k):
        mfilter = np.where(label == i)
        plt.scatter(data[mfilter,0], data[mfilter,1], marker='o', label=int(i), s=5)
    plt.legend(loc='upper left', title="Classes")
    # plt.legend(loc='upper left', title="Classes", ncol=3)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if limit_x != None:
        plt.xlim(limit_x[0], limit_x[1])
    if limit_y != None:
        plt.ylim(limit_y[0], limit_y[1])
    plt.tight_layout()
    plt.savefig('result/'+ name + '-embedding.png')
    if is_show :
        plt.show()


def plot_embedding_3d(name, data, label, k=10, is_show=False):
    ax = plt.figure().gca(projection='3d')
    for i in range(k):
        mfilter = np.where(label == i)
        ax.scatter(data[mfilter,0], data[mfilter,1], data[mfilter,2], marker='o', label=int(i))
    ax.legend(loc='upper left', title="Classes")
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    plt.tight_layout()
    plt.savefig('result/'+ name + '-embedding.png')
    if is_show :
        plt.show()


def plot_pca_2d(name, data, label, k=10, is_show=False):
    data = torch.from_numpy(data)
    data_PCA = find_PCA(data, 2)
    plot_embedding_2d(name, data_PCA, label, k, is_show)


def plot_pca_3d(name, data, label, k=10, is_show=False):
    data = torch.from_numpy(data)
    data_PCA = find_PCA(data, 3)
    plot_embedding_2d(name, data_PCA, label, k, is_show)


def plot_emb_distance(name, data, label, k=10, is_show=False):
    # normalize
    min_data = np.min(data)
    max_data = np.max(data)
    data_range = max_data - min_data
    data = (data-min_data) / data_range
    # norm_data = np.linalg.norm(data, axis=1)
    f = open('output/embedding-'+ name +'-distance.txt',"w")
    for i in range(k):
        print(i)
        intra = []
        filtered = np.where(label == i)
        obj_l = data[filtered]
        # n = len(obj_l)
        n = 500
        for x in range(n-1):
            for y in range(x+1,n):
                temp = np.linalg.norm(obj_l[x] - obj_l[y])
                intra.append(temp)
        intra = np.asarray(intra)
        inter = []
        for j in range(k):
            print(i, j)
            if i != j:
                filtered = np.where(label == i)
                obj_l = data[filtered]
                filtered = np.where(label == j)
                obj_r = data[filtered]
                # n = len(obj_l)
                # m = len(obj_r)
                n = 500
                m = 500
                for x in range(n):
                    for y in range(m):
                        temp = np.linalg.norm(obj_l[x] - obj_r[y])
                        inter.append(temp)
        inter = np.asarray(inter)
        mean_intra = np.mean(intra)
        std_intra = np.std(intra)
        mean_inter = np.mean(inter)
        std_inter = np.std(inter)
        plt.figure()
        plt.hist(inter, bins='auto', alpha = 0.7, label='Intra')
        plt.hist(intra, bins='auto', alpha = 0.7, label='Inter')
        plt.xlim(-0.1, 1.5)
        # plt.ylim(0, 100000)
        plt.xlabel('Distance')
        # plt.xlabel('Distance\nIntra mean:%.2f std:%.2f\nInter mean:%.2f std:%.2f' % (mean_intra, std_intra, mean_inter, std_inter))
        print('%d & %.2f & %.2f & %.2f & %.2f\n' % (i, mean_intra, std_intra, mean_inter, std_inter))
        f.write('%d & %.2f & %.2f & %.2f & %.2f\n' % (i, mean_intra, std_intra, mean_inter, std_inter))
        plt.legend()
        plt.tight_layout()
        plt.savefig('result/' + name + '-embedding-'+str(i)+'.png')
        np.savetxt('result/' + name + '-intra-'+str(i)+'.out',intra)
        np.savetxt('result/' + name + '-inter-'+str(i)+'.out',inter)
    f.close()
    if is_show :
        plt.show()


def plot_cosine(name, data, label, k=10, is_show=False):
    norm_data = np.linalg.norm(data, axis=1)
    sims = []
    for i in range(0,k-1):
        for j in range(i+1, k):
            print(i, j)
            filtered = np.where(label == i)
            obj_l = data[filtered][:200]
            obj_l_norm = norm_data[filtered][:200]
            filtered = np.where(label == j)
            obj_r = data[filtered][:200]
            obj_r_norm = norm_data[filtered][:200]
            n = len(obj_l)
            m = len(obj_r)
            for x in range(n):
                for y in range(m):
                    sim = np.dot(obj_l[x], obj_r[y]) / (obj_l_norm[x] * obj_r_norm[y])
                    sims.append(sim)
    sims = np.asarray(sims)
    plt.figure()
    plt.hist(sims, bins='auto')
    plt.legend(loc='upper left')
    plt.xlim(0, 1)
    plt.savefig('result/' + name + '-embedding.png')
    if is_show :
        plt.show()


def plot_cosine_2(name, data1, label1, data2, label2, k=10, is_show=False):
    def calculate_cosine(data, label):
        min_data = np.min(data)
        max_data = np.max(data)
        data_range = max_data - min_data
        data = (data-min_data) / data_range
        norm_data = np.linalg.norm(data, axis=1)
        a_obj = []
        for i in range(k):
            filtered = np.where(label == i)
            obj_l = data[filtered]
            obj_l_norm = norm_data[filtered]
            n = len(obj_l)
            sims = []
            for x in range(n-1):
                for y in range(x+1,n):
                    sim = np.dot(obj_l[x], obj_l[y]) / (obj_l_norm[x] * obj_l_norm[y])
                    sims.append(sim)
            sims = np.asarray(sims)
            a_obj.append(sims)
        return a_obj
    A = calculate_cosine(data1, label1)
    B = calculate_cosine(data2, label2)
    for i in range(k):
        plt.figure()
        plt.hist([A[i], B[i]], bins='auto', label=['With Analogy Potential','Witout Analogy Potential'])
        plt.legend(loc='upper left')
        plt.xlim(0, 1)
        plt.savefig('result/' + name + '-embedding-' + str(i) + '.png')
    if is_show :
        plt.show()


def save_histogram_figure(p_arr, n_arr, name, is_show=False, limit_x=None, limit_y=None):
    p_arr = np.asarray(p_arr)
    # mean_arr = np.mean(arr)
    # std_arr = np.std(arr)
    plt.figure()
    if n_arr != None:
        plt.hist(p_arr, bins='auto', alpha=0.7, label='intra')
        plt.hist(n_arr, bins='auto', alpha=0.7, label='inter')
    else:
        plt.hist(p_arr, bins='auto', alpha=0.7)
    # plt.xlabel('Distance\nmean:%.2f std:%.2f' % (mean_arr, std_arr))
    if limit_x != None:
        plt.xlim(limit_x[0], limit_x[1])
    if limit_y != None:
        plt.ylim(limit_y[0], limit_y[1])
    plt.xlabel('Distance')
    plt.ylabel('Number of samples')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('result/' + name + '-hist.png')
    if is_show:
        plt.show()
    else:
        plt.close()


def plot_distance(name, data, label, limit_x=None, limit_y=None, k=10, using_average=False, n_length=-1, is_show=False):
    # normalize
    min_data = np.min(data)
    max_data = np.max(data)
    data_range = max_data - min_data
    data = (data-min_data) / data_range
    # norm_data = np.linalg.norm(data, axis=1)
    arr = []
    if using_average:
        func = np.average
    else:
        func = np.linalg.norm
    plt.figure()
    for i in range(k):
        print('process', i, i)
        filtered = np.where(label == i)
        obj_l = data[filtered]
        if n_length == -1:
            n = len(obj_l)
        else:
            n = n_length
        p_arr = []
        for x in range(n-1):
            for y in range(x+1, n):
                temp = func(obj_l[x] - obj_l[y])
                p_arr.append(temp)
        arr = arr + p_arr
        plt.hist(p_arr, bins='auto', alpha=0.5, label=str(i)+'-'+str(i))
        n_arr = []
        for j in range(k):
            print('process', i, j)
            if i != j:
                filtered = np.where(label == i)
                obj_l = data[filtered]
                filtered = np.where(label == j)
                obj_r = data[filtered]
                if n_length == -1:
                    n = len(obj_l)
                    m = len(obj_r)
                else:
                    n = n_length
                    m = n_length
                v_arr = []
                for x in range(n):
                    for y in range(m):
                        temp = func(obj_l[x] - obj_r[y])
                        v_arr.append(temp)
                n_arr = n_arr + v_arr
                if j > i:
                    arr = arr + v_arr
                plt.hist(v_arr, bins='auto', alpha=0.5, label=str(i)+'-'+str(j))
        plt.xlabel('Distance')
        plt.ylabel('Number of samples')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('result/' + name + '-hist-'+str(i)+'-all.png')
        plt.close()

        save_histogram_figure(p_arr, n_arr, name+'-'+str(i), is_show=is_show, limit_x=limit_x, limit_y=limit_y)
    save_histogram_figure(arr, None, name, is_show=is_show)

def plot_distance_v2(name, data, label, label_arr, limit_x=None, limit_y=None, k=10, using_average=False, n_length=-1, is_show=False):
    # normalize
    min_data = np.min(data)
    max_data = np.max(data)
    data_range = max_data - min_data
    data = (data-min_data) / data_range
    # norm_data = np.linalg.norm(data, axis=1)
    if using_average:
        func = np.average
    else:
        func = np.linalg.norm
    plt.figure(figsize=(10, 12))
    n_col = 2
    n_row = k / n_col
    for i in range(k):
        print('process', i, i)
        filtered = np.where(label == i)
        obj_l = data[filtered]
        if n_length == -1:
            n = len(obj_l)
        else:
            n = n_length
        p_arr = []
        for x in range(n-1):
            for y in range(x+1, n):
                temp = func(obj_l[x] - obj_l[y])
                p_arr.append(temp)
        plt.subplot(n_row, n_col, i+1)
        plt.hist(p_arr, bins='auto', alpha=0.5, label='Intra')
        n_arr = []
        for j in range(k):
            print('process', i, j)
            if i != j:
                filtered = np.where(label == i)
                obj_l = data[filtered]
                filtered = np.where(label == j)
                obj_r = data[filtered]
                if n_length == -1:
                    n = len(obj_l)
                    m = len(obj_r)
                else:
                    n = n_length
                    m = n_length
                v_arr = []
                for x in range(n):
                    for y in range(m):
                        temp = func(obj_l[x] - obj_r[y])
                        v_arr.append(temp)
                n_arr = n_arr + v_arr
        plt.hist(n_arr, bins='auto', alpha=0.5, label='Inter')
        if limit_x != None:
            plt.xlim(limit_x[0], limit_x[1])
        if limit_y != None:
            plt.ylim(limit_y[0], limit_y[1])
        plt.xlabel('Distance')
        plt.ylabel('Number of samples')
        plt.title('Histogram of '+label_arr[i])
        plt.legend(loc='upper right')
        plt.tight_layout()
    plt.savefig('result/' + name + '-hist.png')
    plt.close()