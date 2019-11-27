#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : model.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .utils import *


class Value(nn.Module):
    def __init__(
        self, num_inputs, num_outputs=1, hidden_size=(128, 128), activation="tanh"
    ):
        super().__init__()
        if activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = num_inputs
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, num_outputs)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value


class DiscretePolicy(nn.Module):
    def __init__(
        self, state_dim, action_num, hidden_size=(128, 128), activation="tanh"
    ):
        super().__init__()
        self.is_disc_action = True
        if activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_head = nn.Linear(last_dim, action_num)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

    def select_action(self, x, is_train=True):
        action_prob = self.forward(x)
        if is_train:
            action = action_prob.multinomial(1)
        else:
            _, action = action_prob.max(dim=1)
        return action.data

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return action_prob.gather(1, actions.unsqueeze(1))


if __name__ == "__main__":
    pass
