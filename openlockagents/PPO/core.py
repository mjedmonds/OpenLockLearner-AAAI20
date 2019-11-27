#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : core.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import torch
from torch.autograd import Variable

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
    max_iter_num,
    learning_rate,
    clip_epsilon,
    l2_reg,
    i_iter,
    optim_epochs,
    optim_value_epochs,
    optim_batch_size,
):
    states = tensor_type(batch.state)
    actions = action_tensor_type(batch.action)
    rewards = tensor_type(batch.reward)
    masks = tensor_type(batch.mask)
    with torch.no_grad():
        values = value(Variable(states)).data
    fixed_log_probs = policy.get_log_prob(
        Variable(states, volatile=True), Variable(actions)
    ).data

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(
        rewards, masks, values, gamma, epsilon, tensor_type
    )

    lr_mult = max(1.0 - float(i_iter) / max_iter_num, 0)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm.tolist())
        states, actions, returns, advantages, fixed_log_probs = (
            states[perm],
            actions[perm],
            returns[perm],
            advantages[perm],
            fixed_log_probs[perm],
        )

        for i in range(optim_iter_num):
            ind = slice(
                i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0])
            )
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = (
                states[ind],
                actions[ind],
                advantages[ind],
                returns[ind],
                fixed_log_probs[ind],
            )

            ppo_step(
                policy,
                value,
                optimizer_policy,
                optimizer_value,
                optim_value_epochs,
                states_b,
                actions_b,
                returns_b,
                advantages_b,
                fixed_log_probs_b,
                lr_mult,
                learning_rate,
                clip_epsilon,
                l2_reg,
            )


def ppo_step(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    optim_value_iternum,
    states,
    actions,
    returns,
    advantages,
    fixed_log_probs,
    lr_mult,
    lr,
    clip_epsilon,
    l2_reg,
):
    set_lr(optimizer_policy, lr * lr_mult)
    set_lr(optimizer_value, lr * lr_mult)
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = returns.requires_grad_()
    for _ in range(optim_value_iternum):
        values_pred = value_net(states.requires_grad_())
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    advantages_var = advantages.requires_grad_()
    log_probs = policy_net.get_log_prob(states.requires_grad_(), actions)
    ratio = torch.exp(log_probs - fixed_log_probs.requires_grad_())
    surr1 = ratio * advantages_var
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
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
