#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation_node(hidden_layer_activation):
    if hidden_layer_activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif hidden_layer_activation == 'tanh':
        return nn.Tanh()
    elif hidden_layer_activation == 'sigmoid':
        return nn.Sigmoid()
    elif hidden_layer_activation == 'softmax':
        return nn.Softmax(dim=1)
    else:
        return nn.ReLU()

def create_layers(input_nn, output_nn, layer_nn=1, hidden_layer_activation='relu'):
    layers = []
    layers.append(nn.Linear(input_nn, output_nn))
    layers.append(get_activation_node(hidden_layer_activation))
    for k in range(1, layer_nn):
        layers.append(nn.Linear(output_nn, output_nn))
        layers.append(get_activation_node(hidden_layer_activation))
    return nn.Sequential(*layers)

def create_layers_from_arr(input_nn, sequence_arr, hidden_layer_activation='relu'):
    layers = []
    num_layer = len(sequence_arr) # len must be greater than 0
    layers.append(nn.Linear(input_nn, sequence_arr[0]))
    layers.append(get_activation_node(hidden_layer_activation))
    for k in range(1, num_layer):
        layers.append(nn.Linear(sequence_arr[k-1], sequence_arr[k]))
        layers.append(get_activation_node(hidden_layer_activation))
    return nn.Sequential(*layers)

def create_input_layer(input_nn, output_nn, hidden_layer_activation='relu'):
    return nn.Sequential(
                nn.Linear(input_nn, output_nn),
                get_activation_node(hidden_layer_activation)
        )    

def create_output_layer(input_nn):
    return nn.Sequential(
            nn.Linear(input_nn, 1),
            nn.Sigmoid()
        )


# =============
# Analogy Model
# =============

def l1_distance(layer1):
    return torch.abs((layer1[0] - layer1[1]) - (layer1[2] - layer1[3]))

"""
===WARNING!===
   If x is a finite value less than 0, and y is a finite noninteger, 
a domain error occurs, and a NaN is returned. 

   Except as specified below, if x or y is a NaN, the result is a NaN.

   If x is negative, then large negative or positive y values yield a NaN 
as the function result, with  errno  set  to  EDOM,  and  an  invalid
   (FE_INVALID)  floating-point  exception.  For example, with pow(), 
one sees this behavior when the absolute value of y is greater than about
   9.223373e18.
"""

def l2_distance(layer1):
    return torch.pow(torch.abs((layer1[0] - layer1[1]) - (layer1[2] - layer1[3])), 2)

def l1_and_l2_distance(layer1):
    return torch.pow(torch.abs((layer1[0] - layer1[1]) - (layer1[2] - layer1[3]), 2)) + torch.abs((layer1[0] - layer1[1]) - (layer1[2] - layer1[3]))

def l1_and_min_distance(layer1):
    return torch.abs((layer1[0] - layer1[1]) - (layer1[2] - layer1[3])) + torch.min(torch.abs(layer1[0]-layer1[1]), torch.abs(layer1[0]-layer1[2]))

def cosine_similarity(layer1):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return torch.abs(cos((layer1[0] - layer1[1]), (layer1[2] - layer1[3])))
    # return torch.abs(cos(layer1[0],layer1[1]) - cos(layer1[2],layer1[3]))

def get_function(fn_str):
    if fn_str == 'L1':
        return l1_distance
    elif fn_str =='L2':
        return l2_distance
    elif fn_str == 'L1andL2':
        return l1_and_l2_distance
    elif fn_str =='L1andMin':
        return l1_and_min_distance
    elif fn_str =='Cosine':
        return cosine_similarity

class AnalogyModel(nn.Module):
    def __init__(self,
        size=784,
        input_layers=[], 
        decision_layers=[], 
        using_analogy_fn=False,
        using_shared_weight=False,
        act='relu'
    ):
        super(AnalogyModel, self).__init__()
        self.layer = size
        self.input_layers = input_layers
        self.decision_layers = decision_layers
        self.act = act
        self.using_analogy_fn = using_analogy_fn
        self.using_shared_weight = using_shared_weight
        temp_input_nn = self.layer
        
        # Shared weight
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                self.il = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]
        else:
            if len(self.input_layers) > 0:
                self.il1 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il3 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il4 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]

        if self.using_analogy_fn:
            self.func = l1_distance
            if len(self.decision_layers) > 0:
                self.fl = create_layers_from_arr(temp_input_nn, self.decision_layers, self.act)
                temp_input_nn = self.decision_layers[-1]
            self.fc_out = create_output_layer(temp_input_nn)
        else:
            temp_input_nn = temp_input_nn * 4
            if len(self.decision_layers) > 0:
                self.fl = create_layers_from_arr(temp_input_nn, self.decision_layers, self.act)
                temp_input_nn = self.decision_layers[-1]
            self.fc_out = create_output_layer(temp_input_nn)
        

    def forward(self, x1, x2, x3, x4):
        layer1 = [x1, x2, x3, x4]
        for i in range(len(layer1)):
            layer1[i] = layer1[i].view(layer1[i].shape[0], -1)
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                for i in range(len(layer1)):
                    layer1[i] = self.il(layer1[i])
        else:
            if len(self.input_layers) > 0:
                layer1[0] = self.il1(layer1[0])
                layer1[1] = self.il2(layer1[1])
                layer1[2] = self.il3(layer1[2])
                layer1[3] = self.il4(layer1[3])
 
        if self.using_analogy_fn:
            layer2 = self.func(layer1)
        else:
            layer2 = torch.cat((layer1[0], layer1[1], layer1[2], layer1[3]), 1)
        if len(self.decision_layers) > 0:
            layer2 = self.fl(layer2)
        layer3 = self.fc_out(layer2)
        return layer3, layer1[0], layer1[1], layer1[2], layer1[3]


class OldAnalogyModel(nn.Module):
    def __init__(self,
        size=784,
        input_layers=[], 
        first_layers=[], 
        second_layers=[], 
        using_shared_weight=False,
        act='relu',
    ):
        super(OldAnalogyModel, self).__init__()
        self.layer = size,
        self.input_layers = input_layers
        self.first_layers = first_layers
        self.second_layers = second_layers
        self.act = act
        self.using_shared_weight = using_shared_weight
        temp_input_nn = self.layer
        
        # Shared weight
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                self.il1 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]
        else:
            if len(self.input_layers) > 0:
                self.il1 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il3 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il4 = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]
        temp_input_nn = temp_input_nn * 2  
        if self.using_shared_weight:
            if len(self.first_layers) > 0:
                self.fl = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                temp_input_nn = self.first_layers[-1]
        else:
            if len(self.first_layers) > 0:
                self.fl1 = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                self.fl2 = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                temp_input_nn = self.first_layers[-1]
        temp_input_nn = temp_input_nn * 2
        if len(self.second_layers) > 0:
            self.sl = create_layers_from_arr(temp_input_nn, self.second_layers, self.act)
            temp_input_nn = self.second_layers[-1]
        self.fc_out = create_output_layer(temp_input_nn)
        

    def forward(self, x1, x2, x3, x4):
        layer1 = [x1, x2, x3, x4]
        for i in range(len(layer1)):
            layer1[i] = layer1[i].view(layer1[i].shape[0], -1)
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                layer1[0] = self.il1(layer1[0])
                layer1[1] = self.il1(layer1[1])
                layer1[2] = self.il2(layer1[2])
                layer1[3] = self.il2(layer1[3])
        else:
            if len(self.input_layers) > 0:
                layer1[0] = self.il1(layer1[0])
                layer1[1] = self.il2(layer1[1])
                layer1[2] = self.il3(layer1[2])
                layer1[3] = self.il4(layer1[3])
 
        layer2 = [None, None]
        layer2[0] = torch.cat((layer1[0], layer1[1]), 1)
        layer2[1] = torch.cat((layer1[2], layer1[3]), 1)
        if self.using_shared_weight:
            if len(self.first_layers) > 0:
                layer2[0] = self.fl(layer2[0])
                layer2[1] = self.fl(layer2[1])
        else:
            if len(self.first_layers) > 0:
                layer2[0] = self.fl1(layer2[0])
                layer2[1] = self.fl2(layer2[1])
        layer3 = torch.cat((layer2[0], layer2[1]), 1)
        if len(self.second_layers) > 0:
            layer3 = self.sl(layer3)
        layer4 = self.fc_out(layer3)
        return layer4, layer1[0], layer1[1], layer1[2], layer1[3]


class OldAnalogyModelWithExchangeMean(nn.Module):
    def __init__(self,
        size=784,
        input_layers=[], 
        first_layers=[], 
        second_layers=[], 
        third_layers=[], 
        using_shared_weight=False,
        act='relu',
    ):
        super(OldAnalogyModelWithExchangeMean, self).__init__()
        self.layer = 784
        self.input_layers = input_layers
        self.first_layers = first_layers
        self.second_layers = second_layers
        self.third_layers = third_layers
        self.act = act
        self.using_shared_weight = using_shared_weight
        temp_input_nn = self.layer
        
        # Shared weight
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                self.il1_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il1_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]
        else:
            if len(self.input_layers) > 0:
                self.il1_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il3_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il4_l = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il1_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il2_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il3_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                self.il4_r = create_layers_from_arr(self.layer, self.input_layers, self.act)
                temp_input_nn = self.input_layers[-1]
        temp_input_nn = temp_input_nn * 2  
        if self.using_shared_weight:
            if len(self.first_layers) > 0:
                self.fl_l = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                self.fl_r = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                temp_input_nn = self.first_layers[-1]
        else:
            if len(self.first_layers) > 0:
                self.fl1_l = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                self.fl2_l = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                self.fl1_r = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                self.fl2_r = create_layers_from_arr(temp_input_nn, self.first_layers, self.act)
                temp_input_nn = self.first_layers[-1]
        temp_input_nn = temp_input_nn * 2
        if self.using_shared_weight:
            if len(self.second_layers) > 0:
                self.sl_l = create_layers_from_arr(temp_input_nn, self.second_layers, self.act)
                self.sl_r = create_layers_from_arr(temp_input_nn, self.second_layers, self.act)
                temp_input_nn = self.second_layers[-1]
        else:
            if len(self.second_layers) > 0:
                self.sl = create_layers_from_arr(temp_input_nn, self.second_layers, self.act)
                temp_input_nn = self.second_layers[-1]
        temp_input_nn = temp_input_nn * 2
        if len(self.third_layers) > 0:
            self.tl = create_layers_from_arr(temp_input_nn, self.third_layers, self.act)
            temp_input_nn = self.third_layers[-1]
        self.fc_out = create_output_layer(temp_input_nn)
        

    def forward(self, x1, x2, x3, x4):
        layer1 = [x1, x2, x3, x4, x1, x3, x2, x4]
        for i in range(len(layer1)):
            layer1[i] = layer1[i].view(layer1[i].shape[0], -1)
        if self.using_shared_weight:
            if len(self.input_layers) > 0:
                layer1[0] = self.il1_l(layer1[0])
                layer1[1] = self.il1_l(layer1[1])
                layer1[2] = self.il2_l(layer1[2])
                layer1[3] = self.il2_l(layer1[3])
                layer1[4] = self.il1_r(layer1[4])
                layer1[5] = self.il1_r(layer1[5])
                layer1[6] = self.il2_r(layer1[6])
                layer1[7] = self.il2_r(layer1[7])
        else:
            if len(self.input_layers) > 0:
                layer1[0] = self.il1_l(layer1[0])
                layer1[1] = self.il2_l(layer1[1])
                layer1[2] = self.il3_l(layer1[2])
                layer1[3] = self.il4_l(layer1[3])
                layer1[4] = self.il1_r(layer1[4])
                layer1[5] = self.il2_r(layer1[5])
                layer1[6] = self.il3_r(layer1[6])
                layer1[7] = self.il4_r(layer1[7])
 
        layer2 = [None, None, None, None]
        layer2[0] = torch.cat((layer1[0], layer1[1]), 1)
        layer2[1] = torch.cat((layer1[2], layer1[3]), 1)
        layer2[2] = torch.cat((layer1[4], layer1[5]), 1)
        layer2[3] = torch.cat((layer1[6], layer1[7]), 1)
        if self.using_shared_weight:
            if len(self.first_layers) > 0:
                layer2[0] = self.fl_l(layer2[0])
                layer2[1] = self.fl_l(layer2[1])
                layer2[2] = self.fl_r(layer2[2])
                layer2[3] = self.fl_r(layer2[3])
        else:
            if len(self.first_layers) > 0:
                layer2[0] = self.fl1_l(layer2[0])
                layer2[1] = self.fl2_l(layer2[1])
                layer2[2] = self.fl1_r(layer2[2])
                layer2[3] = self.fl2_r(layer2[3])
        layer3 = [None, None]
        layer3[0] = torch.cat((layer2[0], layer2[1]), 1)
        layer3[1] = torch.cat((layer2[2], layer2[3]), 1)
        if self.using_shared_weight:
            if len(self.second_layers) > 0:
                layer3[0] = self.sl_l(layer3[0])
                layer3[1] = self.sl_r(layer3[1])
        else:
            if len(self.second_layers) > 0:
                layer3[0] = self.sl(layer3[0])
                layer3[1] = self.sl(layer3[1])
        layer4 = torch.cat((layer3[0], layer3[1]), 1)
        if len(self.third_layers) > 0:
            layer4 = self.tl(layer4)
        layer5 = self.fc_out(layer4)
        return layer5, layer1[0], layer1[1], layer1[2], layer1[3]


class AnalogyCustomModel(nn.Module):
    def __init__(self, size=784, number_nn=800, number_layer=2, number_output=10, act='relu'):
        super(AnalogyCustomModel, self).__init__()
        self.number_nn = number_nn
        self.number_layer = number_layer
        self.number_output = number_output
        self.layer = size
        self.f1 = nn.Linear(self.layer, self.number_nn)
        layers = []
        for i in range(self.number_layer):
            layers.append(nn.Linear(self.number_nn, self.number_nn))
        self.f2 = nn.Sequential(*layers)
        self.f3 = nn.Linear(self.number_nn, self.number_output)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.out = get_activation_node(act)

        # TEST
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward_once(self, x):
        t = self.f1(x)
        t = self.act1(t)
        t = self.f2(t)
        t = self.act2(t)
        t = self.f3(t)
        t = self.out(t)
        return t

        # TEST
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.softmax(self.fc3(x), dim=1)
        # return x

    def forward(self, x1, x2, x3, x4):
        x = [x1, x2, x3, x4]
        for i in range(len(x)):
            x[i] = x[i].view(x[i].shape[0], -1)
            x[i] = self.forward_once(x[i])
        return None, x[0], x[1], x[2], x[3]