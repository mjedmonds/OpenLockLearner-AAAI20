#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : core.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.optimize

from .utils import *


def update_params(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    params,
    batch_a,
    batch_b,
    i_iter,
):
    """
        batch_a: [num_task, batch_size, ...]
        batch_b: [num_task, batch_size, ...]
    """
    if params["backbone"] == "a2c":
        maml_a2c_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            batch_a,
            batch_b,
            params,
        )
    elif params["backbone"] == "trpo":
        maml_trpo_step(policy_net, value_net, batch_a, batch_b, params)
    elif params["backbone"] == "ppo":
        maml_ppo_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            batch_a,
            batch_b,
            i_iter,
            params,
        )
    else:
        raise NotImplementedError


def update_params_k_shot(
    policy_net, value_net, optimizer_policy, optimizer_value, params, batch, i_iter
):
    states = params["Tensor"](batch.state)
    actions = params["ActionTensor"](batch.action)
    rewards = params["Tensor"](batch.reward)
    masks = params["Tensor"](batch.mask)
    with torch.no_grad():
        values = value_net(states)

    advantages, returns = estimate_advantages(
        rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
    )

    if params["backbone"] == "a2c":
        a2c_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            states,
            actions,
            returns,
            advantages,
            params["l2_reg"],
        )
    elif params["backbone"] == "trpo":
        trpo_step(
            policy_net,
            value_net,
            states,
            actions,
            returns,
            advantages,
            params["max_kl"],
            params["damping"],
            params["l2_reg"],
        )
    elif params["backbone"] == "ppo":
        lr_mult = 1.0  # max(1.0 - float(i_iter) / params['max_iter_num'], 0)
        with torch.no_grad():
            fixed_log_probs = policy_net.get_log_prob(states, actions)
        ppo_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            params["optim_value_epochs"],
            states,
            actions,
            returns,
            advantages,
            fixed_log_probs,
            lr_mult,
            params["learning_rate"],
            params["clip_epsilon"],
            params["l2_reg"],
        )
    else:
        raise NotImplementedError


def maml_a2c_step(
    policy_net, value_net, optimizer_policy, optimizer_value, batch_a, batch_b, params
):
    env_num = len(batch_a)
    init_policy_param = get_flat_params_from(policy_net)
    init_value_param = get_flat_params_from(value_net)
    pre_policy_param = []
    pre_value_param = []
    # pre-update
    set_lr(optimizer_policy, params["lr_pre_update"])
    set_lr(optimizer_value, params["lr_pre_update"])
    for env_id in range(env_num):
        set_flat_params_to(policy_net, init_policy_param)
        set_flat_params_to(value_net, init_value_param)
        batch = batch_a[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        for _ in range(params["num_grad_update"]):
            a2c_step(
                policy_net,
                value_net,
                optimizer_policy,
                optimizer_value,
                states,
                actions,
                returns,
                advantages,
                params["l2_reg"],
            )
        pre_policy_param.append(get_flat_params_from(policy_net))
        pre_value_param.append(get_flat_params_from(value_net))

    # meta-update
    meta_policy_param = []
    meta_value_param = []
    set_lr(optimizer_policy, params["lr_meta_update"])
    set_lr(optimizer_value, params["lr_meta_update"])
    for env_id in range(env_num):
        set_flat_params_to(policy_net, pre_policy_param[env_id])
        set_flat_params_to(value_net, pre_value_param[env_id])
        batch = batch_b[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        a2c_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            states,
            actions,
            returns,
            advantages,
            params["l2_reg"],
        )
        meta_policy_param.append(get_flat_params_from(policy_net))
        meta_value_param.append(get_flat_params_from(value_net))
    # computer mixture of gradients
    for env_id in range(env_num):
        init_policy_param += meta_policy_param[env_id] - pre_policy_param[env_id]
        init_value_param += meta_value_param[env_id] - pre_value_param[env_id]

    set_flat_params_to(policy_net, init_policy_param)
    set_flat_params_to(value_net, init_value_param)


def maml_trpo_step(policy_net, value_net, batch_a, batch_b, params):
    env_num = len(batch_a)
    init_policy_param = get_flat_params_from(policy_net)
    init_value_param = get_flat_params_from(value_net)
    pre_policy_param = []
    pre_value_param = []
    # pre-update
    for env_id in range(env_num):
        set_flat_params_to(policy_net, init_policy_param)
        set_flat_params_to(value_net, init_value_param)
        batch = batch_a[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        for _ in range(params["num_grad_update"]):
            trpo_step(
                policy_net,
                value_net,
                states,
                actions,
                returns,
                advantages,
                params["max_kl"],
                params["damping"],
                params["l2_reg"],
            )
        pre_policy_param.append(get_flat_params_from(policy_net))
        pre_value_param.append(get_flat_params_from(value_net))

    # meta-update
    meta_policy_param = []
    meta_value_param = []
    for env_id in range(env_num):
        set_flat_params_to(policy_net, pre_policy_param[env_id])
        set_flat_params_to(value_net, pre_value_param[env_id])
        batch = batch_b[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        trpo_step(
            policy_net,
            value_net,
            states,
            actions,
            returns,
            advantages,
            params["max_kl"],
            params["damping"],
            params["l2_reg"],
        )
        meta_policy_param.append(get_flat_params_from(policy_net))
        meta_value_param.append(get_flat_params_from(value_net))
    # computer mixture of gradients
    for env_id in range(env_num):
        init_policy_param += meta_policy_param[env_id] - pre_policy_param[env_id]
        init_value_param += meta_value_param[env_id] - pre_value_param[env_id]

    set_flat_params_to(policy_net, init_policy_param)
    set_flat_params_to(value_net, init_value_param)


def maml_ppo_step(
    policy_net,
    value_net,
    optimizer_policy,
    optimizer_value,
    batch_a,
    batch_b,
    i_iter,
    params,
):
    env_num = len(batch_a)
    init_policy_param = get_flat_params_from(policy_net)
    init_value_param = get_flat_params_from(value_net)
    pre_policy_param = []
    pre_value_param = []
    lr_mult = 1.0  # max(1.0 - float(i_iter) / params['max_iter_num'], 0)
    # pre-update
    for env_id in range(env_num):
        set_flat_params_to(policy_net, init_policy_param)
        set_flat_params_to(value_net, init_value_param)
        batch = batch_a[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        for _ in range(params["num_grad_update"]):
            ppo_step(
                policy_net,
                value_net,
                optimizer_policy,
                optimizer_value,
                params["optim_value_epochs"],
                states,
                actions,
                returns,
                advantages,
                fixed_log_probs,
                lr_mult,
                params["lr_pre_update"],
                params["clip_epsilon"],
                params["l2_reg"],
            )
        pre_policy_param.append(get_flat_params_from(policy_net))
        pre_value_param.append(get_flat_params_from(value_net))

    # meta-update
    meta_policy_param = []
    meta_value_param = []
    for env_id in range(env_num):
        set_flat_params_to(policy_net, pre_policy_param[env_id])
        set_flat_params_to(value_net, pre_value_param[env_id])
        batch = batch_b[env_id]
        states = params["Tensor"](batch.state)
        actions = params["ActionTensor"](batch.action)
        rewards = params["Tensor"](batch.reward)
        masks = params["Tensor"](batch.mask)
        with torch.no_grad():
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions)
        advantages, returns = estimate_advantages(
            rewards, masks, values, params["gamma"], params["epsilon"], params["Tensor"]
        )
        ppo_step(
            policy_net,
            value_net,
            optimizer_policy,
            optimizer_value,
            params["optim_value_epochs"],
            states,
            actions,
            returns,
            advantages,
            fixed_log_probs,
            lr_mult,
            params["lr_meta_update"],
            params["clip_epsilon"],
            params["l2_reg"],
        )
        meta_policy_param.append(get_flat_params_from(policy_net))
        meta_value_param.append(get_flat_params_from(value_net))
    # computer mixture of gradients
    for env_id in range(env_num):
        init_policy_param += meta_policy_param[env_id] - pre_policy_param[env_id]
        init_value_param += meta_value_param[env_id] - pre_value_param[env_id]

    set_flat_params_to(policy_net, init_policy_param)
    set_flat_params_to(value_net, init_value_param)


def conjugate_gradients(Avp_f, b, nsteps, rdotr_tol=1e-10):
    x = zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp = Avp_f(p)
        alpha = rdotr / torch.dot(p, Avp)
        x += alpha * p
        r -= alpha * Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < rdotr_tol:
            break
    return x


def line_search(
    model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1
):
    fval = f(True).item()

    for stepfrac in [0.5 ** x for x in range(max_backtracks)]:
        x_new = x + stepfrac * fullstep
        set_flat_params_to(model, x_new)
        fval_new = f(True).item()
        actual_improve = fval - fval_new
        expected_improve = expected_improve_full * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio:
            return True, x_new
    return False, x


def trpo_step(
    policy_net, value_net, states, actions, returns, advantages, max_kl, damping, l2_reg
):
    """update critic"""
    values_target = returns.requires_grad_()

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states.requires_grad_())
        value_loss = (values_pred - values_target).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.cpu().item(), get_flat_grad_from(value_net).cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss, get_flat_params_from(value_net).cpu().numpy(), maxiter=25
    )
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    """update policy"""
    with torch.no_grad():
        fixed_log_probs = policy_net.get_log_prob(states, actions)
    """define the loss function for TRPO"""

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                log_probs = policy_net.get_log_prob(states, actions)
        else:
            log_probs = policy_net.get_log_prob(states, actions)

        action_loss = -advantages.requires_grad_() * torch.exp(
            log_probs - fixed_log_probs.requires_grad_()
        )
        return action_loss.mean()

    """define Hessian*vector for KL"""

    def Fvp(v):
        kl = policy_net.get_kl(states.requires_grad_())
        kl = kl.mean()

        grads = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * v.detach().requires_grad_()).sum()
        grads = torch.autograd.grad(kl_v, policy_net.parameters())
        flat_grad_grad_kl = torch.cat(
            [grad.contiguous().view(-1) for grad in grads]
        ).data

        return flat_grad_grad_kl + v * damping

    loss = get_loss()
    grads = torch.autograd.grad(loss, policy_net.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir.dot(Fvp(stepdir)))
    lm = math.sqrt(max_kl / shs)
    fullstep = stepdir * lm
    expected_improve = -loss_grad.dot(fullstep)

    prev_params = get_flat_params_from(policy_net)
    success, new_params = line_search(
        policy_net, get_loss, prev_params, fullstep, expected_improve
    )
    set_flat_params_to(policy_net, new_params)

    return success


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
