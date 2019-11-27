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


class QDiscretePolicy(nn.Module):
    def __init__(self, q_net, eps_start=0.9, eps_end=0.05, eps_decay=10):
        super().__init__()
        self.is_disc_action = True
        self.q_net = q_net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = float(eps_decay)
        self.iter = 0

    def forward(self, x):
        value = self.q_net(x)
        return value

    def select_action(self, x, is_train=True):
        # \epsilon-greedy
        value = self.forward(x)
        if is_train:
            eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1 * self.iter / self.eps_decay
            )
        else:
            eps = 0
        if np.random.rand() < eps:
            return torch.randint(0, value.shape[-1], (value.shape[0],))
        else:
            return torch.argmax(value, dim=-1)


if __name__ == "__main__":
    pass
