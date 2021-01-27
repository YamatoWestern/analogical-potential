#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'My custom dataset loader'

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms
from analogymodel.constant_var import positive_permute, negative_permute


class LoadByPathAnalogyDataset(Dataset):
    def __init__(self, file_path, transform=None, reload_dataset=True, counter=1, constructor=None, deconstructor=None):
        self.file_path = file_path
        self.reload_dataset = reload_dataset
        self.counter = counter
        self.reload()
        self.transform = transform
        self.constructor = constructor
        self.deconstructor = deconstructor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _index):
        index = int(_index)
        if self.constructor == None:
            img1  = Image.open(self.imgs1[index])
            img2  = Image.open(self.imgs2[index])
            img3  = Image.open(self.imgs3[index])
            img4  = Image.open(self.imgs4[index])
        else:
            img1  = self.constructor(self.imgs1[index])
            img2  = self.constructor(self.imgs2[index])
            img3  = self.constructor(self.imgs3[index])
            img4  = self.constructor(self.imgs4[index])
        label1 = int(self.labels1[index])
        label2 = int(self.labels2[index])
        label3 = int(self.labels3[index])
        label4 = int(self.labels4[index])
        label = self.labels[index]
        if self.transform is not None:
            nimg1 = self.transform(img1)
            nimg2 = self.transform(img2)
            nimg3 = self.transform(img3)
            nimg4 = self.transform(img4)
        else:
            nimg1 = torch.from_numpy(img1)
            nimg2 = torch.from_numpy(img2)
            nimg3 = torch.from_numpy(img3)
            nimg4 = torch.from_numpy(img4)
        if self.deconstructor == None:
            img1.close()
            img2.close()
            img3.close()
            img4.close()
            del img1, img2, img3, img4
        else:
            self.deconstructor(img1)
            self.deconstructor(img2)
            self.deconstructor(img3)
            self.deconstructor(img4)
        return [nimg1, nimg2, nimg3, nimg4], [label1, label2, label3, label4], label

    def reload(self):
        if self.reload_dataset:
            self.data = pd.read_csv(self.file_path)
        else:
            file_name = self.file_path % self.counter
            self.data = pd.read_csv(file_name)
            self.counter = self.counter + 1
        self.imgs1  = np.asarray(self.data.iloc[:, 0])
        self.imgs2  = np.asarray(self.data.iloc[:, 1])
        self.imgs3  = np.asarray(self.data.iloc[:, 2])
        self.imgs4  = np.asarray(self.data.iloc[:, 3])
        self.labels1 = np.asarray(self.data.iloc[:, 4])
        self.labels2 = np.asarray(self.data.iloc[:, 5])
        self.labels3 = np.asarray(self.data.iloc[:, 6])
        self.labels4 = np.asarray(self.data.iloc[:, 7])
        self.labels = np.asarray(self.data.iloc[:, 8])


class LoadByPathRawDataset(Dataset):
    def __init__(self, file_path, transform=None, constructor=None, deconstructor=None):
        self.data = pd.read_csv(file_path)
        self.labels1 = np.asarray(self.data.iloc[:, 1])
        self.transform = transform
        self.constructor = constructor
        self.deconstructor = deconstructor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _index):
        index = int(_index)
        if self.constructor == None:
            img1 = Image.open(self.data.iloc[index, 0])
        else:
            img1 = self.constructor(self.data.iloc[index, 0])
        label1 = int(self.labels1[index])
        if self.transform is not None:
            nimg1 = self.transform(img1)
        else:
            nimg1 = torch.from_numpy(img1)
        if self.deconstructor == None:
            img1.close()
        else:
            self.deconstructor(img1)
        del img1
        return nimg1, label1, self.data.iloc[index, 0]


class LoadByPathPairDataset(Dataset):
    def __init__(self, file_path, transform=None, constructor=None, deconstructor=None):
        self.data = pd.read_csv(file_path)
        self.labels1 = np.asarray(self.data.iloc[:, 2])
        self.labels2 = np.asarray(self.data.iloc[:, 3])
        self.transform = transform
        self.constructor = constructor
        self.deconstructor = deconstructor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _index):
        index = int(_index)
        if self.constructor == None:
            img1  = Image.open(self.data.iloc[index, 0])
            img2  = Image.open(self.data.iloc[index, 1])
        else:
            img1 = self.constructor(self.data.iloc[index, 0])
            img2 = self.constructor(self.data.iloc[index, 1])
        label1 = int(self.labels1[index])
        label2 = int(self.labels2[index])
        if self.transform is not None:
            nimg1 = self.transform(img1)
            nimg2 = self.transform(img2)
        else:
            nimg1 = torch.from_numpy(img1)
            nimg2 = torch.from_numpy(img2)
        if self.deconstructor == None:
            img1.close()
            img2.close()
        else:
            self.deconstructor(img1)
            self.deconstructor(img2)
        del img1
        del img2
        return [nimg1, nimg2], [label1, label2]


number_of_permute = 8 # length of positive permute or negative permute. both should be equal
number_of_answer = 2 # {postive, negative}
from itertools import combinations 
import random

from scipy.misc import comb


class LoadByPathAllAnalogyDataset(Dataset):
    def __init__(self, file_path, number_of_class=10, number_of_img=3000, transform=None, constructor=None, deconstructor=None, percent=1., 
        same_class_percent=0.,
        different_class_percent_3=0., different_class_percent_4=0.,
        no_permute=False,
    ):
        self.data = pd.read_csv(file_path)
        self.imgs = np.asarray(self.data.iloc[:, 0])
        self.labels = np.asarray(self.data.iloc[:, 1])
        self.no_permute = no_permute
        self.number_of_permute = number_of_permute
        if self.no_permute:
            self.number_of_permute = 1
        self.balance_length = int(comb(number_of_class, 2) * self.number_of_permute * number_of_answer) # for the balanced dataset
        self.total_balance_length = int(self.balance_length * percent)
        self.number_of_class = number_of_class
        self.list_of_combination = list(combinations([i for i in range(number_of_class)], 2))

        # Unbalace Datset
        self.total_same_class_length = 0
        self.total_different_class_length_3 = 0
        self.total_different_class_length_4 = 0
        self.same_class_length = 0
        self.different_class_length_3 = 0
        self.different_class_length_4 = 0
        self.using_same_class = False
        self.using_different_class = False
        if same_class_percent > 0:
            self.using_same_class = True
            self.same_class_length = int(number_of_class * self.number_of_permute)
            self.total_same_class_length = int(self.same_class_length * same_class_percent)
        if different_class_percent_3 > 0 or different_class_percent_4 > 0:
            self.using_different_class = True
            self.different_class_length_3 = int(comb(number_of_class, 3) * self.number_of_permute) # for the balanced dataset
            self.different_class_length_4 = int(comb(number_of_class, 4) * self.number_of_permute) # for the balanced dataset
            self.total_different_class_length_3 = int(self.different_class_length_3 * different_class_percent_3)
            self.total_different_class_length_4 = int(self.different_class_length_4 * different_class_percent_4)
            self.list_of_combination_3 = list(combinations([i for i in range(number_of_class)], 3))
            self.list_of_combination_4 = list(combinations([i for i in range(number_of_class)], 4))
            self.number_of_combination_3 = len(self.list_of_combination_3)
            self.number_of_combination_4 = len(self.list_of_combination_4)

        self.number_of_answer = number_of_answer
        self.number_of_combination = len(self.list_of_combination)
        self.number_of_permute = len(positive_permute) # assume that positive permute equals to negative permute 
        self.transform = transform
        self.number_of_img = number_of_img
        self.pool = None
        self.pool_same_class = None
        self.pool_different_class_3 = None
        self.pool_different_class_4 = None
        self.iter = 0
        self.iter_same_class = 0
        self.iter_different_class_3 = 0
        self.iter_different_class_4 = 0
        self.total_length = self.total_balance_length + self.total_different_class_length_3 + self.total_different_class_length_4 + self.total_same_class_length
        self.reload()
        print(self.balance_length, self.total_balance_length)
        print(self.same_class_length, self.total_same_class_length)
        print(self.different_class_length_3, self.total_different_class_length_3)
        print(self.different_class_length_4, self.total_different_class_length_4)

        self.constructor = constructor
        self.deconstructor = deconstructor

    def get_index(self, class_idx):
        start_range = class_idx * self.number_of_img
        stop_range = start_range + self.number_of_img - 1
        return random.randint(start_range, stop_range)

    def reload(self):
        total_length = self.balance_length if self.balance_length > self.total_balance_length else self.total_balance_length
        self.pool = [ i for i in range(self.total_balance_length) ]
        self.iter = self.total_balance_length
        if self.total_same_class_length > 0:
            total_length = self.same_class_length if self.same_class_length > self.total_same_class_length else self.total_same_class_length
            self.pool_same_class = [ i for i in range(total_length) ]
            self.iter_same_class = self.total_same_class_length
        if self.total_different_class_length_3 > 0:
            total_length = self.different_class_length_3 if self.different_class_length_3 > self.total_different_class_length_3 else self.total_different_class_length_3
            self.pool_different_class_3 = [ i for i in range(total_length) ]
            self.iter_different_class_3 = self.total_different_class_length_3
        if self.total_different_class_length_4 > 0:
            total_length = self.different_class_length_4 if self.different_class_length_4 > self.total_different_class_length_4 else self.total_different_class_length_4
            self.pool_different_class_4 = [ i for i in range(total_length) ]
            self.iter_different_class_4 = self.total_different_class_length_4

    def __len__(self):
        return self.total_length

    def get_balance_index(self, _index):
        p_idx = random.randint(0, self.iter-1)
        index = self.pool[p_idx]
        self.pool[p_idx] = self.pool[self.iter-1]
        self.iter = self.iter-1

        t_index = index % self.number_of_answer # For decide it positive or negative
        index_2 = index // self.number_of_answer
        if self.no_permute:
            p_index = random.randint(0,7)
            index_3 = index_2
        else:
            p_index = index_2 % self.number_of_permute
            index_3 = index_2 // self.number_of_permute
        if t_index == 0:
            permute = positive_permute[p_index]
        elif t_index == 1:
            permute = negative_permute[p_index]
        c_index = index_3 % self.number_of_combination
        first_class = self.list_of_combination[c_index][0]
        second_class = self.list_of_combination[c_index][1]
        idx = [
                self.get_index(first_class),
                self.get_index(second_class),
                self.get_index(first_class),
                self.get_index(second_class)
            ]
        index1 = idx[permute[0]]
        index2 = idx[permute[1]]
        index3 = idx[permute[2]]
        index4 = idx[permute[3]]
        return t_index, index1, index2, index3, index4

    def get_unbalance_index_same_class(self, _index):
        p_idx = random.randint(0, self.iter_same_class-1)
        index = self.pool_same_class[p_idx]
        self.pool_same_class[p_idx] = self.pool_same_class[self.iter_same_class-1]
        self.iter_same_class = self.iter_same_class-1

        # same class
        if self.no_permute:
            p_index = random.randint(0,7)
            index_2 = index
        else:
            p_index = index % self.number_of_permute
            index_2 = index // self.number_of_permute
        permute = positive_permute[p_index]
        c_index = index_2 % self.number_of_class
        first_class = c_index
        idx = [
            self.get_index(first_class),
            self.get_index(first_class),
            self.get_index(first_class),
            self.get_index(first_class)
        ]
        index1 = idx[permute[0]]
        index2 = idx[permute[1]]
        index3 = idx[permute[2]]
        index4 = idx[permute[3]]
        return 0, index1, index2, index3, index4
        
    def get_unbalance_index_different_class_3(self, _index):
        p_idx = random.randint(0, self.iter_different_class_3-1)
        index = self.pool_different_class_3[p_idx]
        self.pool_different_class_3[p_idx] = self.pool_different_class_3[self.iter_different_class_3-1]
        self.iter_different_class_3 = self.iter_different_class_3-1
        
        # 1 pair of the same class and 2 different classes
        if self.no_permute:
            p_index = random.randint(0,7)
            index_2 = index
        else:
            p_index = index % self.number_of_permute
            index_2 = index // self.number_of_permute
        permute = positive_permute[p_index]
        c_index = index_2 % self.number_of_combination_3
        first_class = self.list_of_combination_3[c_index][0]
        second_class = self.list_of_combination_3[c_index][1]
        third_class = self.list_of_combination_3[c_index][2]
        idx = [
            self.get_index(first_class),
            self.get_index(first_class),
            self.get_index(second_class),
            self.get_index(third_class)
        ]
        permute = positive_permute[p_index]
        index1 = idx[permute[0]]
        index2 = idx[permute[1]]
        index3 = idx[permute[2]]
        index4 = idx[permute[3]]
        return 1, index1, index2, index3, index4

    def get_unbalance_index_different_class_4(self, _index):
        p_idx = random.randint(0, self.iter_different_class_4-1)
        index = self.pool_different_class_4[p_idx]
        self.pool_different_class_4[p_idx] = self.pool_different_class_4[self.iter_different_class_4-1]
        self.iter_different_class_4 = self.iter_different_class_4-1

        # different class
        if self.no_permute:
            p_index = random.randint(0,7)
            index_2 = index
        else:
            p_index = index % self.number_of_permute
            index_2 = index // self.number_of_permute
        permute = positive_permute[p_index]
        c_index = index_2 % self.number_of_combination_4
        first_class = self.list_of_combination_4[c_index][0]
        second_class = self.list_of_combination_4[c_index][1]
        third_class = self.list_of_combination_4[c_index][2]
        fourth_class = self.list_of_combination_4[c_index][3]
        idx = [
            self.get_index(first_class),
            self.get_index(second_class),
            self.get_index(third_class),
            self.get_index(fourth_class)
        ]
        permute = positive_permute[p_index]
        index1 = idx[permute[0]]
        index2 = idx[permute[1]]
        index3 = idx[permute[2]]
        index4 = idx[permute[3]]
        return 1, index1, index2, index3, index4

    def __getitem__(self, _index):
        if _index < self.total_balance_length :
            t_index, index1, index2, index3, index4 = self.get_balance_index(_index)
        elif self.using_same_class and (_index - self.total_balance_length) < self.total_same_class_length :
            temp_index = _index - self.total_balance_length
            t_index, index1, index2, index3, index4 = self.get_unbalance_index_same_class(temp_index)
        elif self.using_different_class :
            temp_index =  _index - self.total_balance_length - self.total_same_class_length
            if temp_index < self.total_different_class_length_3:
                t_index, index1, index2, index3, index4 = self.get_unbalance_index_different_class_3(temp_index)
            else:
                temp_index = temp_index - self.total_different_class_length_3
                t_index, index1, index2, index3, index4 = self.get_unbalance_index_different_class_4(temp_index)
        if self.constructor == None:
            img1  = Image.open(self.imgs[index1])
            img2  = Image.open(self.imgs[index2])
            img3  = Image.open(self.imgs[index3])
            img4  = Image.open(self.imgs[index4])
        else:
            img1 = self.constructor(self.imgs[index1])
            img2 = self.constructor(self.imgs[index2])
            img3 = self.constructor(self.imgs[index3])
            img4 = self.constructor(self.imgs[index4])
        label1 = int(self.labels[index1])
        label2 = int(self.labels[index2])
        label3 = int(self.labels[index3])
        label4 = int(self.labels[index4])
        if self.transform is not None:
            nimg1 = self.transform(img1)
            nimg2 = self.transform(img2)
            nimg3 = self.transform(img3)
            nimg4 = self.transform(img4)
        else:
            nimg1 = torch.from_numpy(img1)
            nimg2 = torch.from_numpy(img2)
            nimg3 = torch.from_numpy(img3)
            nimg4 = torch.from_numpy(img4)
        if self.deconstructor == None:
            img1.close()
            img2.close()
            img3.close()
            img4.close()
        else:
            self.deconstructor(img1)
            self.deconstructor(img2)
            self.deconstructor(img3)
            self.deconstructor(img4)
        del img1
        del img2
        del img3
        del img4
        # print(_index, label1, label2, label3, label4, t_index)
        return [nimg1, nimg2, nimg3, nimg4], [label1, label2, label3, label4], t_index
