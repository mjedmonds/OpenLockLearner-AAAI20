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
    policy,
    value,
    optimizer_policy,
    optimizer_value,
    batch,
    tensor_type,
    action_tensor_type,
    gamma,
    epsilon,
    l2_reg,
):
    states = tensor_type(batch.state)
    actions = action_tensor_type(batch.action)
    rewards = tensor_type(batch.reward)
    masks = tensor_type(batch.mask)
    with torch.no_grad():
        values = value(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(
        rewards, masks, values, gamma, epsilon, tensor_type
    )

    """perform TRPO update"""
    a2c_step(
        policy,
        value,
        optimizer_policy,
        optimizer_value,
        states,
        actions,
        returns,
        advantages,
        l2_reg,
    )


def a2c_step(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    returns,
    advantages,
    l2_reg,
):
    """update critic"""
    values_target = returns.requires_grad_()
    values_pred = value_net(states.requires_grad_())
    value_loss = F.mse_loss(values_pred, values_target)
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states.requires_grad_(), actions)
    policy_loss = -(log_probs * advantages.requires_grad_()).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()


def estimate_advantages(rewards, masks, values, gamma, epsilon, tensor_type):
    returns = tensor_type(rewards.size(0), 1)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * epsilon * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
    advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns


if __name__ == "__main__":
    pass
