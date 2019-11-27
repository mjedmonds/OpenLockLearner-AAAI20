#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import sys
import os
import atexit
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt

import torch

from openlock.settings_trial import PARAMS, IDX_TO_PARAMS
from openlock.settings_scenario import select_scenario
from openlock.envs.openlock_env import ObservationSpace

from openlockagents.common import ROOT_DIR
from openlockagents.MAML.k_shot_agent import MAML_K_Shot_Agent


def create_reward_fig():
    # creating the figure
    # plt.ion()
    fig = plt.figure()
    fig.set_size_inches(20, 5)
    # plt.pause(0.0001)
    return fig


def main():
    torch.set_default_tensor_type("torch.DoubleTensor")

    # general params
    # training params
    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS["CE3-CE4"]
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS['CC4']
    else:
        setting = sys.argv[1]
        params = PARAMS[IDX_TO_PARAMS[int(setting) - 1]]
        print(
            "training_scenario: {}, testing_scenario: {}".format(
                params["train_scenario_name"], params["test_scenario_name"]
            )
        )
        params["reward_mode"] = sys.argv[2]

    # a2c
    params["epsilon"] = 0.95
    params["l2_reg"] = 1e-3
    # trpo
    params["max_kl"] = 1e-2
    params["damping"] = 1e-2
    # ppo
    params["clip_epsilon"] = 0.2
    params["optim_value_epochs"] = 1
    # maml
    params["backbone"] = "trpo"

    # generic
    params["learning_rate"] = 0.01
    params["batch_size"] = 2048
    params["gamma"] = 0.99
    params["reward_mode"] = "basic"
    params["use_gpu"] = True
    params["gpuid"] = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

    params["Tensor"] = (
        torch.cuda.DoubleTensor if params["use_gpu"] else torch.DoubleTensor
    )
    params["ActionTensor"] = (
        torch.cuda.LongTensor if params["use_gpu"] else torch.LongTensor
    )

    random_seed = 1234
    params["use_physics"] = False
    params[
        "full_attempt_limit"
    ] = (
        False
    )  # run to the full attempt limit, regardless of whether or not all solutions were found
    params["num_training_iters"] = r00
    params["num_training_trials"] = params["train_num_trials"]
    params["train_attempt_limit"] = 700

    params["num_testing_iters"] = r00
    params["num_testing_trials"] = params["test_num_trials"]
    params["test_attempt_limit"] = 700

    # RL specific settings
    params["data_dir"] = os.path.dirname(ROOT_DIR) + "/OpenLockRLResults/subjects"

    scenario = select_scenario(
        params["train_scenario_name"], use_physics=params["use_physics"]
    )

    env = gym.make("openlock-v1")
    env.use_physics = params["use_physics"]
    env.full_attempt_limit = params["full_attempt_limit"]
    # set up observation space
    env.observation_space = ObservationSpace(
        len(scenario.levers), append_solutions_remaining=False
    )
    # set reward mode
    env.reward_mode = params["reward_mode"]
    print("Reward mode: {}".format(env.reward_mode))
    np.random.seed(random_seed)
    env.seed(random_seed)

    # dummy agent
    agent = MAML_K_Shot_Agent(env, 1, 1, params, require_log=False)
    trial_selected = agent.setup_trial(
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
    )
    env.reset()

    state_size = agent.env.observation_space.multi_discrete.shape[0]
    action_size = len(env.action_space)
    agent = MAML_K_Shot_Agent(env, state_size, action_size, params)
    save_path = os.path.join(
        params["data_dir"],
        "3rd_model_log/k_shot-{}-{}-{}".format(
            params["train_scenario_name"], params["reward_mode"], agent.subject_id
        ),
    )
    # save_path = os.path.join(params['data_dir'], '3rd_model_log/k_shot-CC3-{}-{}-{}'.format(
    #                                    params['train_scenario_name'], params['reward_mode'],
    #                                    agent.subject_id))
    load_path = sys.argv[3] if len(sys.argv) >= 4 else ""  # path without '.*' suffix
    os.makedirs(save_path, exist_ok=True)

    agent.env.reset()
    if load_path:
        agent.load(load_path)
        print("load model from {}".format(load_path))
    else:
        print("[Warn] No meta-trained model found, will transfer from scratch")

    agent.env.human_agent = False
    agent.type_tag = "{}-K_Shot".format(params["train_scenario_name"])
    # train over multiple iterations over all trials
    fig = create_reward_fig()
    for iter_num in range(params["num_training_iters"]):
        agent.env.completed_trials = []
        for trial_num in range(0, params["num_training_trials"]):
            agent.run_trial_maml_k_shot(
                scenario_name=params["train_scenario_name"],
                fig=fig,
                action_limit=params["train_action_limit"],
                attempt_limit=params["train_attempt_limit"],
                trial_count=trial_num,
                iter_num=iter_num,
            )
            fig, data = agent.log_values(
                [
                    agent.trial_length,
                    agent.trial_percent_attempt_success,
                    agent.trial_percent_solution_found,
                    agent.average_trial_rewards,
                    agent.attempt_rewards,
                ],
                fig,
                [
                    "Attempt Count Per Trial",
                    "Percentage of Successful Attempts in Trial",
                    "Percentage of Solutions Found in Trial",
                    "Average Trial Reward",
                    "Attempt Reward",
                ],
                agent.type_tag,
            )
            pickle.dump(
                (agent.type_tag, data, params),
                open(os.path.join(save_path, "log.pkl"), "wb"),
            )
            # update
            if len(agent.memory) > params["batch_size"]:
                batch = agent.memory.sample()
                print("update with bs:{}".format(len(batch.state)))
                agent.update(batch, iter_num)
                agent.memory.clear()
        agent.save(save_path, iter_num)
    print(
        "Trial complete for subject {}. Average reward: {}".format(
            agent.logger.subject_id, agent.average_trial_rewards[-1]
        )
    )
    fig.savefig(os.path.join(save_path, "log.png"))


if __name__ == "__main__":
    main()
