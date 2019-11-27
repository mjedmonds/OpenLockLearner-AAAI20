#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : core.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .utils import *


def update_params(
    q_net,
    target_q_net,
    optimizer,
    batch,
    tensor_type,
    action_tensor_type,
    gamma,
    l2_reg,
):
    states = tensor_type(batch.state)
    actions = action_tensor_type(batch.action)
    rewards = tensor_type(batch.reward)
    next_states = tensor_type(batch.next_state)
    masks = tensor_type(batch.mask)

    """perform TRPO update"""
    dqn_step(
        q_net,
        target_q_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        masks,
        gamma,
        l2_reg,
    )


def dqn_step(
    q_net,
    target_q_net,
    optimizer,
    states,
    actions,
    rewards,
    next_states,
    masks,
    gamma,
    l2_reg,
):
    """update Q"""
    masks = masks.to(torch.uint8)
    values_pred = q_net(states.requires_grad_()).gather(
        -1, actions.unsqueeze(-1).long()
    )  # N*1
    values_target = torch.zeros_like(values_pred).squeeze()
    values_target[masks] = target_q_net(next_states[masks]).max(1)[0].detach()
    values_target = values_target * gamma + rewards

    value_loss = F.smooth_l1_loss(values_pred, values_target.unsqueeze(-1))
    # weight decay
    for param in q_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer.zero_grad()
    value_loss.backward()
    optimizer.step()


if __name__ == "__main__":
    pass
