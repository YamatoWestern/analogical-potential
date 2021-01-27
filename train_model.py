#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'Run this script for train/test the model'

import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib
matplotlib.use('Agg') # For save the figure without showing
import numpy as np

# My functions
from analogymodel.custom_function import init_parser, init_log, support_old_format, type_embedding, get_criteria_fn
from analogymodel.custom_mnist_dataset import LoadByPathAllAnalogyDataset, LoadByPathAnalogyDataset, LoadByPathRawDataset
from analogymodel.custom_loss import *
from analogymodel.model_analogy import *
from analogymodel.custom_ploter import plot_sample, plot_graph, plot_embedding
# For plot 3D graph
from mpl_toolkits.mplot3d import Axes3D 
# debug
from torch import autograd


def create_sample_results(args, model, device, data_loader):
    model = model.cpu() # Move back to CPU
    examples = enumerate(data_loader)
    batch_idx, (imgs, _, target) = next(examples)
    with torch.no_grad():
        output, _, _, _, _ = model(imgs[0], imgs[1], imgs[2], imgs[3])
        pics = []
        titles = []
        for i in range(6):
            pics.append(torch.cat((imgs[0][i], imgs[1][i], imgs[2][i], imgs[3][i]), 2).view(28, 112))
            titles.append("Truth:{} Answer:{}".format(int(target[i]), int(output[i].round())))
        plot_sample(args.custom_name, pics, titles)


def export_embedding(args, model, data_loader, tname='test'):
    model = model.cpu() # move back to cpu
    model_name = args.custom_name
    if args.load_model != '':
        model_name = args.load_model
    model.load_state_dict(torch.load('model/'+model_name+'.pt'))
    model.eval()
    with torch.no_grad():
        labels = []
        with open('embedding/'+args.custom_name+'-'+tname+'-embedding.txt',"w") as f:
            for batch_idx, (img, label, _) in enumerate(data_loader):
                _, x1, x2, x3, x4 = model(img, img, img, img)
                for i in range(len(x1)):
                    for j in range(len(x1[i])):
                        f.write('%f ' % (x1[i,j]))
                    f.write('\n')
                    labels.append(label[i])
        with open('embedding/'+args.custom_name+'-'+tname+'-label.txt',"w") as f:
            for i in range(len(labels)):
                f.write('%d\n' % (labels[i]))


import matplotlib.pyplot as plt

def train(args, model, device, train_loader, optimizer, epoch, is_display):
    model.train()
    total_loss = 0
    for batch_idx, (imgs, labels, gt) in enumerate(train_loader):
        with autograd.detect_anomaly():
            if args.using_contrastive_loss:
                target = gt.type(torch.FloatTensor).view(gt.shape[0], 1)
                target = target.view(-1)
                imgs[0], imgs[1], imgs[2], imgs[3], target = imgs[0].to(device), imgs[1].to(device), imgs[2].to(device), imgs[3].to(device), target.to(device)
                optimizer.zero_grad()
                _, A, B, C, D = model(imgs[0], imgs[1], imgs[2], imgs[3])
                embs = [A, B, C, D]
                loss = criteria(embs, target)
                total_loss += loss.item()
                sum_loss = loss
                if sub_criteria != None:
                    sub_loss = sub_criteria(embs, target)
                    total_loss += sub_loss.item()
                    sum_loss += sub_loss
                if sub_criteria_2 != None:
                    sub_loss = sub_criteria_2(embs, target)
                    total_loss += sub_loss.item()
                    sum_loss += sub_loss
            else:
                target = gt.type(torch.FloatTensor).view(gt.shape[0], 1)
                imgs[0], imgs[1], imgs[2], imgs[3], target = imgs[0].to(device), imgs[1].to(device), imgs[2].to(device), imgs[3].to(device), target.to(device)
                optimizer.zero_grad()
                output, emb_a, emb_b, emb_c, emb_d = model(imgs[0], imgs[1], imgs[2], imgs[3])
                loss = criteria(output, target)
                total_loss += loss.item()
                sum_loss = loss
            sum_loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logging.info('[Epoch {}/{}] [Batch {}/{}] [loss: {:.6f}]'.format(
                    epoch, args.epoch, batch_idx, len(train_loader), sum_loss.item()
                    ))
    total_loss /= len(train_loader)
    logging.info('[Epoch {}/{}] [loss: {:.6f}]'.format(epoch, args.epoch, total_loss))
    return total_loss


def test(args, model, device, test_loader, is_display, validate=True, save_output=False):
    # save output = save the answer
    model.eval()
    total_loss = 0
    correct = 0
    if save_output:
        f = open('answer/'+args.custom_name+'-answer.txt',"w")
    with torch.no_grad():
        for imgs, labels, gt in test_loader:
            if args.using_contrastive_loss:
                target = gt.type(torch.FloatTensor).view(gt.shape[0], 1)
                target = target.view(-1)
                imgs[0], imgs[1], imgs[2], imgs[3], target = imgs[0].to(device), imgs[1].to(device), imgs[2].to(device), imgs[3].to(device), target.to(device)
                _, A, B, C, D = model(imgs[0], imgs[1], imgs[2], imgs[3])
                embs = [A, B, C, D]
                loss = criteria(embs, target)
                total_loss += loss
                if sub_criteria != None:
                    sub_loss = sub_criteria(embs, target)
                    total_loss += sub_loss
                if sub_criteria_2 != None:
                    sub_loss = sub_criteria_2(embs, target)
                    total_loss += sub_loss
                output = torch.clamp(torch.sum(arithmetic_analogy_function(embs), dim=1)/4, min=0.0, max=1.0)
            else:
                target = gt.type(torch.FloatTensor).view(gt.shape[0], 1)
                imgs[0], imgs[1], imgs[2], imgs[3], target = imgs[0].to(device), imgs[1].to(device), imgs[2].to(device), imgs[3].to(device), target.to(device)
                output, emb_a, emb_b, emb_c, emb_d = model(imgs[0], imgs[1], imgs[2], imgs[3])
                total_loss += criteria(output, target).item()
            pred = output.round()
            if save_output :
                for i in range(len(pred)):
                    f.write('%d\n' % (pred[i]))
            correct += pred.eq(target.view_as(pred)).sum().item()
    if save_output:
        f.close()
    total_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    if validate:
        logging.info('[Validate] [Avg. loss: {:.6f}] [Accuracy: {}/{} ({:.2f}%)]'.format(
            total_loss, correct, len(test_loader.dataset), acc))
    else:
        logging.info('[Test] [Avg. loss: {:.6f}] [Accuracy: {}/{} ({:.2f}%)]'.format(
            total_loss, correct, len(test_loader.dataset), acc))
    return total_loss, acc


if __name__ == '__main__':
    # parsing the arguments from user
    parser = argparse.ArgumentParser(description='PyTorch Analogy Check')
    parser = init_parser(parser)
    args = parser.parse_args()
    support_old_format(args)

    # init log, seed and cuda usage
    init_log(args.custom_name)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # using transform to transform the data you can modify your own transform here
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    if args.normalize :
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.img_size), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((args.normalize_mean,), (args.normalize_std,))
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # select between fix dataset mean always the same dataset
    # you have to pre-generated them
    # file_path, number_of_class, number_of_sample, number_of_img
    if args.fix_dataset:
        # load fixed dataset
        training_dataset = LoadByPathAnalogyDataset(file_path=args.train_file, transform=transform, reload_dataset=False)
        develop_dataset = LoadByPathAnalogyDataset(file_path=args.dev_file, transform=transform)
        testing_dataset = LoadByPathAnalogyDataset(file_path=args.test_file, transform=transform)
    else:
        training_dataset = LoadByPathAllAnalogyDataset(file_path=args.train_file, 
            number_of_class=args.number_of_class, number_of_img=args.number_of_image, 
            transform=transform, percent=args.percent,
            same_class_percent=args.same_class_percent,
            different_class_percent_3=args.different_class_percent_3, different_class_percent_4=args.different_class_percent_4,
            no_permute=args.no_permute,
            )
        develop_dataset = LoadByPathAnalogyDataset(file_path=args.dev_file, transform=transform)
        testing_dataset = LoadByPathAnalogyDataset(file_path=args.test_file, transform=transform)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(develop_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=args.test_batch_size, shuffle=False)

    # support old format
    input_layers = []
    decision_layers = []
    if args.input_layers != '':
        input_layers = [int(l) for l in args.input_layers.split(",")]
    if args.decision_layers != '':
        decision_layers = [int(l) for l in args.decision_layers.split(",")]

    # old format
    if args.using_old_model:
        input_layers = []
        first_layers = []
        second_layers = []
        input_layers = [args.input_nn] * args.input_layer
        first_layers = [args.first_nn] * args.first_layer
        second_layers = [args.second_nn] * args.second_layer
        third_layers = [args.third_nn] * args.third_layer
    else:
        if args.input_layer is not None:
            input_layers = [args.lower_nn] * args.input_layer
        if args.first_layer is not None:
            decision_layers = [args.upper_nn] * args.first_layer

    # specific the model
    # old_model = hierarchical networks
    # contrastive loss = to enable contrastive loss network
    if args.using_old_model :
        if args.using_exchange_mean :
            model = OldAnalogyModelWithExchangeMean(
                size=args.size,
                input_layers=input_layers,
                first_layers=first_layers,
                second_layers=second_layers,
                third_layers=third_layers,
                act=args.act,
                using_shared_weight=args.using_shared_weight
            )
        else:
            model = OldAnalogyModel(
                size=args.size,
                input_layers=input_layers,
                first_layers=first_layers,
                second_layers=second_layers,
                act=args.act,
                using_shared_weight=args.using_shared_weight
            )
    elif args.using_contrastive_loss :
        model = AnalogyCustomModel(
            size=args.size,
            number_nn=args.number_nn,
            number_layer=args.number_layer,
            number_output=args.number_output,
            act=args.act
        )
    else:
        model = AnalogyModel(
            size=args.size,
            input_layers=input_layers,
            decision_layers=decision_layers,
            act=args.act,
            using_analogy_fn=args.using_analogy_fn,
            using_shared_weight=args.using_shared_weight
            )
    
    # you can enable multiple GPU here
    if torch.cuda.device_count() > 1:
        print('This machine support multiple GPUs')
        if args.enable_multiple_gpus:
            logging.info('Using multiple GPUs')
            model = nn.DataParallel(model)
    model = model.to(device)

    logging.info(args)
    logging.info(model)

    # select the optimizer
    if args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for contrastive loss you can have multi-task learning you can modify it here
    if args.using_contrastive_loss :
        criteria = AnalogyPotentialLoss()
        sub_criteria = None
        sub_criteria_2 = None

        if args.version == 1 :
            criteria = AnalogyPotentialLoss()
        elif args.version == 2:
            criteria = AnalogyPotentialLoss_v2()
        elif args.version == 3:
            criteria = AnalogyPotentialLoss()
            sub_criteria = AnalogyRatioLoss_v3()
        elif args.version == 4:
            criteria = AnalogyPotentialLoss_v2()
            sub_criteria = AnalogyRatioLoss_v3()
        elif args.version == 5:
            criteria = AnalogyPotentialLoss()
            sub_criteria = AnalogyRatioLoss_v4()
        elif args.version == 6:
            criteria = AnalogyPotentialLoss_v2()
            sub_criteria = AnalogyRatioLoss_v4()
        elif args.version == 7:
            criteria = AnalogyPotentialLoss()
            sub_criteria = AnalogyRatioLoss_v3()
            sub_criteria_2 = AnalogyDistanceLoss()
        elif args.version == 8:
            criteria = AnalogyPotentialLoss_v2()
            sub_criteria = AnalogyRatioLoss_v3()
            sub_criteria_2 = AnalogyDistanceLoss()
        elif args.version == 9:
            criteria = AnalogyPotentialLoss()
            sub_criteria = AnalogyDistanceLoss()
        elif args.version == 10:
            criteria = AnalogyPotentialLoss_v2()
            sub_criteria = AnalogyDistanceLoss()
    else:
        # loss function like BCE, MSE
        criteria = get_criteria_fn(args.criteria)

    # set the save model name
    if args.save_model == '':
        args.save_model = args.custom_name

    # if you load the model it just test your system
    if args.load_model != '':
        model.load_state_dict(torch.load('model/'+args.load_model+'.pt'))
        test(args, model, device, test_loader, args.display, False, args.save_file)
        # create_sample_results(args, model, device, test_loader)
    else:
        # Init
        train_loss = []
        dev_loss = []
        accuracies = []
        axis = []
        axis_dev = []
        test_loss = []
        test_accuracies = []
        min_data = 100
        stop_counter = 0
        for epoch in range(1, args.epoch + 1):
            train_loss_data = train(args, model, device, train_loader, optimizer, epoch, args.display)
            if epoch % args.dev_epoch == 0:
                dev_loss_data, acc_data = test(args, model, device, dev_loader, args.display)
                comp_data = dev_loss_data if args.check_loss else (100-acc_data)
                # this line used for check that the dev accuracy/loss is better then we will save and test
                if epoch == 1 or comp_data < min_data :
                    stop_counter = 0
                    min_data = comp_data
                    logging.info('[ === Test === ]')
                    test_loss_data, test_acc_data = test(args, model, device, test_loader, args.display, False, args.save_file)
                    name = ('model/'+args.save_model+'.pt')
                    torch.save(model.state_dict(), name)
                else:
                    stop_counter = stop_counter + 1
                # Resource for Chart
                axis_dev.append(epoch)
                dev_loss.append(dev_loss_data)
                accuracies.append(acc_data)
            # enable this to save the model every N epoches
            if args.save_epoch_model :
                if epoch % args.save_epoch_model == 0:
                    logging.info('[ === Save Epoch Model === ]')
                    name = ('model/'+args.save_model+'-'+str(epoch)+'.pt')
                    torch.save(model.state_dict(), name)
            axis.append(epoch)
            train_loss.append(train_loss_data)

            # special function for reload the dataset
            training_dataset.reload()
            if stop_counter >= args.stop_epoch:
                logging.info('[ === BREAK === ]')
                break

        # Create Chart
        logging.info('[RESULT] [Avg. loss: {:.6f}] [Accuracy: {:.2f}%]\n'.format(test_loss_data, test_acc_data))

        # Dump loss
        if args.save_loss :
            f = open("output/loss-"+args.custom_name+"-train.txt", "w")
            for l in train_loss:
                f.write('{:.6f}\n'.format(l))
            f.close()
            f = open("output/loss-"+args.custom_name+"-dev.txt", "w")
            for l in dev_loss:
                f.write('{:.6f}\n'.format(l))
            f.close()

        # Write Result
        f = open("log/"+args.custom_name+"-result.txt", "a+")
        f.write('[RESULT] [Avg. loss: {:.6f}] [Accuracy: {:.2f}%]\n'.format(test_loss_data, test_acc_data))
        f.close()
        # create_sample_results(args, model, device, test_loader)
        data = [[axis, train_loss], [axis_dev, dev_loss]]
        plot_graph(args.custom_name + "-loss", "Loss", ['Train Set', 'Development Set'], 
            data, y_title='Loss', x_title='Epoch', loc='upper right', xlim = (1, args.epoch), ylim = None, grid=True)
        data = [[axis_dev, accuracies]]
        plot_graph(args.custom_name + "-acc", "Percentage of Accuracy", ['Development Set'], 
            data, y_title='Accuracy', x_title='Epoch', loc='lower right', xlim = (1, args.epoch), ylim = (0, 100), grid=True)

    # this will write the output as the embedding .txt and label .txt file
    # you can use this output for create the t-SNE or get the recognition score
    if args.export_embedding:
        # train
        embedding_testing_dataset = LoadByPathRawDataset(file_path=args.embedding_train_file, transform=transform)
        embedding_test_loader = torch.utils.data.DataLoader(embedding_testing_dataset, batch_size=args.test_batch_size, shuffle=False)
        export_embedding(args, model, embedding_test_loader, 'train')
        # test
        embedding_testing_dataset = LoadByPathRawDataset(file_path=args.embedding_test_file, transform=transform)
        embedding_test_loader = torch.utils.data.DataLoader(embedding_testing_dataset, batch_size=args.test_batch_size, shuffle=False)
        export_embedding(args, model, embedding_test_loader, 'test')