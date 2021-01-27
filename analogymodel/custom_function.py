#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'My custom functions'

import argparse, logging

import numpy as np
import torch


def init_parser(parser):
    # config
    parser.add_argument("--config",
                    help="read config file", metavar="FILE")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for  (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--epoch', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
    parser.add_argument('--dev-epoch', type=int, default=5,
                    help='run validate set every dev-epoch')
    parser.add_argument('--stop-epoch', type=int, default=20,
                    help='stop after no better result than n epoches (default: 20)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', type=str, default='',
                    help='save the model for the best')
    parser.add_argument('--save-epoch-model',  nargs='?', const=1, type=int,
                    help='save the model for every x epoch')
    parser.add_argument('--load-model', type=str, default='',
                    help='load the model')
    parser.add_argument('--display', action='store_true', default=False,
                    help='display sample output')
    parser.add_argument('--permute', action='store_true', default=False,
                    help='using permute when generate the dataset')
    parser.add_argument('--no-permute', action='store_true', default=False,
                    help='not cover all permute when generate the dataset')
    # optimizer
    parser.add_argument('--opt', type=str, default='Adam', metavar='OPT',
                   help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                   help='learning rate (default: 1e-3)')
    parser.add_argument("--b1", type=float, default=0.9, metavar='B1',
                    help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, metavar='B2',
                    help="adam: decay of first order momentum of gradient")
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                    help='apply weight decay for regularization')
    # normalize
    parser.add_argument("--size", type=int, default=784,
                    help="size of input layer")
    parser.add_argument("--img-size", type=int, default=28,
                    help="size of each image dimension [Deprecated]")
    parser.add_argument("--img-channel", type=int, default=1,
                    help="set image channel [Deprecated]")
    parser.add_argument('--normalize', action='store_true', default=False,
                    help='normalize the input (default: None)')
    parser.add_argument("--normalize-mean", type=float, default=0.5,
                    help="set mean for normalize a image")
    parser.add_argument("--normalize-std", type=float, default=0.5,
                    help="set std for normalize a image")
    # dataset
    parser.add_argument('--train-file', type=str, default='mnist-train-3000.csv',
                    help='input file for training (default: mnist-train-3000.csv)')
    parser.add_argument('--test-file', type=str, default='mnist-test.csv',
                    help='input file for testing (default: mnist-test.csv)')
    parser.add_argument('--dev-file', type=str, default='mnist-dev.csv',
                    help='input file for validating (default: mnist-dev.csv)')
    # parser.add_argument('--test-file', type=str, default='mnist-test-500.csv',
    #                 help='input file for testing (default: mnist-test-500.csv)')
    # parser.add_argument('--dev-file', type=str, default='mnist-dev-1000.csv',
    #                 help='input file for validating (default: mnist-dev-1000.csv)')
    parser.add_argument('--fix-dataset', action='store_true', default=False,
                    help='using fixed dataset (default: False)')
    parser.add_argument('--custom-name', type=str, default='analogy',
                    help='custom name using for many purpose')
    parser.add_argument('--train-counter', type=int, default=1,
                    help='set the starter index of the trainging dataset')
    # model
    parser.add_argument('--type', nargs='?', const=1, type=int,
                    help='choosing type of the model [Deprecated]') 
    parser.add_argument('--input-layer', nargs='?', const=1, type=int,
                    help='custom number of input layers [Deprecated]')
    parser.add_argument('--first-layer', nargs='?', const=1, type=int,
                    help='custom number of first layers [Deprecated]')
    parser.add_argument('--second-layer', nargs='?', const=1, type=int,
                    help='custom number of third layers [Deprecated]')
    parser.add_argument('--third-layer', nargs='?', const=1, type=int,
                    help='custom number of second layers [Deprecated]')
    parser.add_argument('--upper-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in upper layers [Deprecated]')
    parser.add_argument('--lower-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in lower layers [Deprecated]')
    parser.add_argument('--input-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in input layers [Deprecated]')
    parser.add_argument('--first-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in first layers [Deprecated]')
    parser.add_argument('--second-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in second layers [Deprecated]')
    parser.add_argument('--third-nn', type=int, default=128,
                    help='setup number of neurons foreach layer in third layers [Deprecated]')
    parser.add_argument('--input-layers', type=str, default='128,128,128',
                    help='custom layers for input module')
    parser.add_argument('--decision-layers', type=str, default='64',
                    help='custom layers for decision module')
    parser.add_argument("--margin", type=float, default=2.0,
                    help="margin for the model (default: 2.0)")
    # old fashion model
    parser.add_argument('--using-old-model', action='store_true', default=False,
                    help='using old model (default: False)')
    parser.add_argument('--first-layers', type=str, default='64',
                    help='custom layers for first layer module')
    parser.add_argument('--second-layers', type=str, default='64',
                    help='custom layers for second layer module')
    parser.add_argument('--third-layers', type=str, default='64',
                    help='custom layers for second layer module')
    # others
    parser.add_argument('--number-nn', type=int, default=800,
                    help='custom number of neural foreach layer')
    parser.add_argument('--number-layer', type=int, default=2,
                    help='custom number of neural layer')
    parser.add_argument('--number-output', type=int, default=10,
                    help='custom number of neural for output layer')
    parser.add_argument('--number-of-image', type=int, default=3000,
                    help='custom number of image in the dataset')
    parser.add_argument('--number-of-class', type=int, default=10,
                    help='custom number of class in the dataset')
    parser.add_argument('--number-of-class-dev', type=int, default=10,
                    help='custom number of class in the dataset')
    parser.add_argument('--number-of-class-test', type=int, default=10,
                    help='custom number of class in the dataset')
    parser.add_argument('--function', type=str, default='L1',
                    help='choosing the energy function L1, L2, L1andL2, L1andMin (default: L1)')
    parser.add_argument('--act', type=str, default='relu',
                    help='choosing the activation function relu, tanh, sigmoid, leakyrelu (default: relu)')
    parser.add_argument('--using-analogy-fn', action='store_true', default=False,
                    help='enable analogy potential (default: False)')
    parser.add_argument('--using-shared-weight', action='store_true', default=False,
                    help='enable shared weight (default: False)')
    parser.add_argument('--using-exchange-mean', action='store_true', default=False,
                    help='enable exchange of mean (default: False)')
    parser.add_argument('--using-contrastive-loss', action='store_true', default=False,
                    help='enable contrastive loss (default: False)')
    parser.add_argument('--criteria', type=str, default='BCE',
                    help='choose the criteria function for the model (default: BCE)')
    # extra
    parser.add_argument('--embedding-train-file', type=str, default='mnist-train-all.csv',
                    help='input file for training the embbeding (default: mnist-train-all.csv)')
    parser.add_argument('--embedding-test-file', type=str, default='mnist-test-all.csv',
                    help='input file for testing the embbeding (default: mnist-test-all.csv)')
    parser.add_argument('--export-embedding', action='store_true', default=False,
                    help='export embedding file (default: False)')
    parser.add_argument('--check-loss', action='store_true', default=False,
                    help='compare loss instead of accuracy (default: False)')
    parser.add_argument('--save-loss', nargs='?', const=1, type=str,
                    help='export the loss value as the file (default: None)')
    parser.add_argument('--save-file', action='store_true', default=False,
                    help='export the predict as the file (default: False)')
    parser.add_argument('--version', type=int, default=1,
                    help='set the version of the model')
    parser.add_argument('--percent', type=float, default=1.00,
                    help='set percentage of the dataset')
    parser.add_argument('--same-class-percent', type=float, default=0.00,
                    help='set percentage of the same class for dataset')
    parser.add_argument('--different-class-percent-3', type=float, default=0.00,
                    help='set percentage of the different 3 classes for dataset')
    parser.add_argument('--different-class-percent-4', type=float, default=0.00,
                    help='set percentage of the different 4 classes for dataset')
    # Cuda
    parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
    parser.add_argument('--enable-multiple-gpus', action='store_true', default=False,
                    help='enable multiple gpu training')
    return parser

def init_config(args, configs):
    args.batch_size = int(configs['batch_size'])
    args.test_batch_size = int(configs['test_batch_size'])
    args.epoch = int(configs['epochs'])
    args.opt = configs['opt']
    args.lr = float(configs['lr'])
    args.b1 = float(configs['b1'])
    args.b2 = float(configs['b2'])
    args.margin = float(configs['margin'])
    args.img_size = int(configs['img_size'])
    args.normalize = bool(configs['normalize'])
    args.normalize_mean = float(configs['normalize_mean'])
    args.normalize_std = float(configs['normalize_std'])
    args.momentum = float(configs['momentum'])
    args.no_cuda = bool(configs['no_cuda'])
    args.seed = int(configs['seed'])
    args.log_interval = int(configs['log_interval'])
    args.save_model = configs['save_model']
    args.save_epcoh_model = configs['save_epoch_model']
    args.load_model = configs['load_model']
    args.permute = bool(configs['permute'])
    args.type = int(configs['type'])
    args.permute = bool(configs['permute'])
    args.use_dev = bool(configs['use_dev'])
    args.number_nn = int(configs['number_nn'])


def init_log(name):
    logFormatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler("log/"+name+".log", mode='a')
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def support_old_format(args):
    """
    Old format:
    type 1 = Linear Model
    type 2 = Fully Connected Model
    type 3 = Fully Connected Model + Weight Sharing
    type 4 = Analogy Function
    type 5 = Analogy Function + Weight Sharing
    """
    if args.type:
        # for old format compatible
        if args.type == 1:
            args.input_layer = 0
            args.first_layer = 0
            args.using_shared_weight = False
            args.using_analogy_fn = False
        if args.type == 2:
            args.using_shared_weight = False
            args.using_analogy_fn = False
        if args.type == 3:
            args.using_shared_weight = True
            args.using_analogy_fn = False
        if args.type == 4:
            args.using_shared_weight = False
            args.using_analogy_fn = True
        if args.type == 5:
            args.using_shared_weight = True
            args.using_analogy_fn = True


def to_categorical(y, num_columns=10, tensor_type=torch.FloatTensor):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0
    return tensor_type(y_cat)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# Type
def type_embedding(val):
    ival = int(val)
    if (ival < 1 or ival > 3):
        raise argparse.ArgumentTypeError("%s is not allow. This argument receives only 2 and 3" % val)
    return ival


def get_criteria_fn(name):
    if name == 'BCE':
        return torch.nn.BCELoss()
    elif name == 'L1':
        return torch.nn.L1Loss()
    elif name == 'MSE':
        return torch.nn.MSELoss()
    elif name == 'COSINE':
        return torch.nn.CosineEmbeddingLoss()
    return None


def find_PCA(data, k=2):
    # preprocess the data
    data_mean = torch.mean(data,0)
    data = data - data_mean.expand_as(data)
    # single value decomposition
    U,S,V = torch.svd(torch.t(data))
    return torch.mm(data,U[:,:k])


def find_centroid(input_data, input_label):
    obj = []
    for i in range(10):
        mfilter = np.where(input_label == i)
        dat = input_data[mfilter]
        obj.append(np.average(dat, axis=0))
    return np.asarray(obj)