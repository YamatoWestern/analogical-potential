#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'LAROCHAIRANGSI Peerapan <peerapan@akane.waseda.jp>'
__date__, __version__ = '28/01/2020', '1.0'
__description__ = 'My custom loss functions'

import torch
import torch.nn
import torch.nn.functional as F


def difference_function(a, b):
    return a - b

def entropy_function(x, y):
    return -1 * y * torch.log(x)


# absolute distance
def arithmetic_analogy_function(inputs):
    return torch.abs((inputs[0] - inputs[1]) - (inputs[2] - inputs[3]))

def arithmetic_analogy_function_v2(inputs):
    return torch.abs((inputs[0] + inputs[3]) - (inputs[1] + inputs[2]))

# euclidean distance
def arithmetic_analogy_function_with_euclidean_distance(inputs):
    return torch.dist(inputs[0] - inputs[1], inputs[2] - inputs[3])

def absolute_distance(a, b):
    return torch.abs(a-b)
# torch.dist for euclidean distance

# contrastive loss
def function_analogy_potential_loss(inputs, target, reduction='mean', fn=arithmetic_analogy_function, margin=4.0, get_output = False):
    distances = torch.sum(fn(inputs), dim=1)
    losses = ((1-target) * distances) + (target * torch.clamp(margin - distances, min=0.0))
    if get_output:
        return losses
    else:
        return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)

def function_analogy_potential_loss_v2(inputs, target, reduction='mean', fn=arithmetic_analogy_function, margin=4.0, eps=1e-9, get_output = False):
    distances = torch.sum(fn(inputs), dim=1)
    losses = ((1-target) * distances) + (target * torch.clamp(margin - distances, min=0.0))
    clamp_losses = torch.clamp(losses/margin, max=1)
    log_losses = -1 * torch.log(torch.clamp(1-clamp_losses, min=eps, max=1.0))
    if get_output:
        return losses
    else:
        return torch.mean(log_losses) if reduction == 'mean' else torch.sum(log_losses)

def function_analogy_distance_loss(inputs, target, reduction='mean', get_output = False):
    AB = torch.abs(inputs[0] - inputs[1])
    CD = torch.abs(inputs[2] - inputs[3])
    losses = torch.sum(torch.abs(AB-CD), dim=1)
    if get_output:
        return losses
    else:
        return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)

# def function_analogy_ratio_loss(inputs, target, reduction='mean', eps=1e-9, factor=0.125):
#     AB = (torch.pow(inputs[0] - inputs[1], 2).sum(1) + eps).sqrt()
#     AC = (torch.pow(inputs[0] - inputs[2], 2).sum(1) + eps).sqrt()
#     DB = (torch.pow(inputs[3] - inputs[1], 2).sum(1) + eps).sqrt()
#     DC = (torch.pow(inputs[3] - inputs[2], 2).sum(1) + eps).sqrt()
#     BA = (torch.pow(inputs[1] - inputs[0], 2).sum(1) + eps).sqrt()
#     BD = (torch.pow(inputs[1] - inputs[3], 2).sum(1) + eps).sqrt()
#     CA = (torch.pow(inputs[2] - inputs[0], 2).sum(1) + eps).sqrt()
#     CD = (torch.pow(inputs[2] - inputs[3], 2).sum(1) + eps).sqrt()
#     term_AB = AB / (AB + AC)
#     loss_AB = -1 * (term_AB) * torch.log(term_AB)
#     term_AC = AC / (AB + AC)
#     loss_AC = -1 * (term_AC) * torch.log(term_AC)
#     term_DB = DB / (DB + DC)
#     loss_DB = -1 * (term_DB) * torch.log(term_DB)
#     term_DC = DC / (DB + DC)
#     loss_DC = -1 * (term_DC) * torch.log(term_DC)
#     term_BA = BA / (BA + BD)
#     loss_BA = -1 * (term_BA) * torch.log(term_BA)
#     term_BD = BD / (BA + BD)
#     loss_BD = -1 * (term_BD) * torch.log(term_BD)
#     term_CA = CA / (CA + CD)
#     loss_CA = -1 * (term_CA) * torch.log(term_CA)
#     term_CD = CD / (CA + CD)
#     loss_CD = -1 * (term_CD) * torch.log(term_CD)
#     losses = (1-target) * factor * (loss_AB + loss_AC + loss_DB + loss_DC + loss_BA + loss_BD + loss_CA + loss_CD)
#     return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


# def function_analogy_ratio_loss_v2(inputs, target, reduction='mean', fn=absolute_distance, margin=1.0, factor=0.25):
#     AB = fn(inputs[0], inputs[1])
#     AC = fn(inputs[0], inputs[2])
#     DB = fn(inputs[3], inputs[1])
#     DC = fn(inputs[3], inputs[2])
#     BA = fn(inputs[1], inputs[0])
#     BD = fn(inputs[1], inputs[3])
#     CA = fn(inputs[2], inputs[0])
#     CD = fn(inputs[2], inputs[3])
#     loss_1 = (2 * AB * AC) - AB - AC + margin
#     loss_2 = (2 * DB * DC) - DB - DC + margin
#     loss_3 = (2 * BA * BD) - BA - BD + margin
#     loss_4 = (2 * CA * CD) - CA - CD + margin
#     losses = factor * (loss_1 + loss_2 + loss_3 + loss_4)
#     return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def function_analogy_ratio_loss_v3(inputs, target, reduction='mean', fn=absolute_distance, margin=2.0, factor=0.25):
    AB = fn(inputs[0], inputs[1])
    AC = fn(inputs[0], inputs[2])
    BA = fn(inputs[1], inputs[0])
    BD = fn(inputs[1], inputs[3])
    CA = fn(inputs[2], inputs[0])
    CD = fn(inputs[2], inputs[3])
    DB = fn(inputs[3], inputs[1])
    DC = fn(inputs[3], inputs[2])
    AB_AC = torch.sum(torch.abs(AB - AC), dim=1)
    BA_BD = torch.sum(torch.abs(BA - BD), dim=1)
    CA_CD = torch.sum(torch.abs(CA - CD), dim=1)
    DB_DC = torch.sum(torch.abs(DB - DC), dim=1)
    loss_1 = ((target) * AB_AC) + ((1-target) * torch.clamp(margin - AB_AC, min=0.0)) 
    loss_2 = ((target) * BA_BD) + ((1-target) * torch.clamp(margin - BA_BD, min=0.0)) 
    loss_3 = ((target) * CA_CD) + ((1-target) * torch.clamp(margin - CA_CD, min=0.0)) 
    loss_4 = ((target) * DB_DC) + ((1-target) * torch.clamp(margin - DB_DC, min=0.0)) 
    losses = factor * (loss_1 + loss_2 + loss_3 + loss_4)
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


def function_analogy_ratio_loss_v4(inputs, target, reduction='mean', fn=absolute_distance, margin=2.0, factor=0.25):
    eps=1e-9
    AB = fn(inputs[0], inputs[1])
    AC = fn(inputs[0], inputs[2])
    BA = fn(inputs[1], inputs[0])
    BD = fn(inputs[1], inputs[3])
    CA = fn(inputs[2], inputs[0])
    CD = fn(inputs[2], inputs[3])
    DB = fn(inputs[3], inputs[1])
    DC = fn(inputs[3], inputs[2])
    AB_AC = torch.sum(torch.abs(AB - AC), dim=1)
    BA_BD = torch.sum(torch.abs(BA - BD), dim=1)
    CA_CD = torch.sum(torch.abs(CA - CD), dim=1)
    DB_DC = torch.sum(torch.abs(DB - DC), dim=1)
    AB_AC = ((target) * AB_AC) + (1-target * torch.clamp(margin - AB_AC, min=0.0))
    BA_BD = ((target) * BA_BD) + (1-target * torch.clamp(margin - BA_BD, min=0.0))
    CA_CD = ((target) * CA_CD) + (1-target * torch.clamp(margin - CA_CD, min=0.0))
    DB_DC = ((target) * DB_DC) + (1-target * torch.clamp(margin - DB_DC, min=0.0))
    AB_AC = torch.clamp(AB_AC/margin, max=1)
    BA_BD = torch.clamp(BA_BD/margin, max=1)
    CA_CD = torch.clamp(CA_CD/margin, max=1)
    DB_DC = torch.clamp(DB_DC/margin, max=1)
    loss_1 = -1 * torch.log(torch.clamp(1-AB_AC, min=eps, max=1.0))
    loss_2 = -1 * torch.log(torch.clamp(1-BA_BD, min=eps, max=1.0))
    loss_3 = -1 * torch.log(torch.clamp(1-CA_CD, min=eps, max=1.0))
    loss_4 = -1 * torch.log(torch.clamp(1-DB_DC, min=eps, max=1.0))
    losses = factor * (loss_1 + loss_2 + loss_3 + loss_4)
    return torch.mean(losses) if reduction == 'mean' else torch.sum(losses)


class AnalogyPotentialLoss(torch.nn.Module):
    def __init__(self, reduction='mean', margin=4.0, fn=arithmetic_analogy_function):
        super(AnalogyPotentialLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.fn = fn

    def forward(self, inputs, target, size_average=True):
        return function_analogy_potential_loss(inputs, target, reduction=self.reduction, margin=self.margin, fn=self.fn)


class AnalogyPotentialLoss_v2(torch.nn.Module):
    def __init__(self, reduction='mean', margin=4.0, fn=arithmetic_analogy_function):
        super(AnalogyPotentialLoss_v2, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.fn = fn

    def forward(self, inputs, target, size_average=True):
        return function_analogy_potential_loss_v2(inputs, target, reduction=self.reduction, margin=self.margin, fn=self.fn)


class AnalogyDistanceLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(AnalogyDistanceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, target, size_average=True):
        return function_analogy_distance_loss(inputs, target, reduction=self.reduction)


# class AnalogyRatioLoss(torch.nn.Module):
#     def __init__(self, reduction='mean'):
#         super(AnalogyRatioLoss, self).__init__()
#         self.reduction = reduction
#         self.eps = 1e-9
#         self.factor = 0.125

#     def forward(self, inputs, target, size_average=True):
#         return function_analogy_ratio_loss(inputs, target, reduction=self.reduction, eps=self.eps, factor=self.factor)


# class AnalogyRatioLoss_v2(torch.nn.Module):
#     def __init__(self, reduction='mean'):
#         super(AnalogyRatioLoss_v2, self).__init__()
#         self.reduction = reduction
#         self.eps = 1e-9
#         self.factor = 0.25

#     def forward(self, inputs, target, size_average=True):
#         return function_analogy_ratio_loss_v2(inputs, target, reduction=self.reduction, factor=self.factor)

class AnalogyRatioLoss_v3(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(AnalogyRatioLoss_v3, self).__init__()
        self.reduction = reduction
        self.eps = 1e-9
        self.factor = 0.25

    def forward(self, inputs, target, size_average=True):
        return function_analogy_ratio_loss_v3(inputs, target, reduction=self.reduction, factor=self.factor)

class AnalogyRatioLoss_v4(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(AnalogyRatioLoss_v4, self).__init__()
        self.reduction = reduction
        self.eps = 1e-9
        self.factor = 0.25

    def forward(self, inputs, target, size_average=True):
        return function_analogy_ratio_loss_v4(inputs, target, reduction=self.reduction, factor=self.factor)

