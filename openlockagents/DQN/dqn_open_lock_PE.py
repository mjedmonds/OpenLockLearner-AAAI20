#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import sys
import os
import pickle
import atexit
from distutils.dir_util import copy_tree
from matplotlib import pyplot as plt

import torch

from openlock.settings_trial import PARAMS, IDX_TO_PARAMS
from openlock.settings_scenario import select_scenario
from openlock.envs.openlock_env import ObservationSpace

from openlockagents.common import ROOT_DIR
from openlockagents.DQN.dqn_agent import DQNAgent


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

    params["prioritized_replay"] = True
    params["max_mem_size"] = 10000
    params["eps_start"] = 0.90
    params["eps_end"] = 0.05
    params["eps_decay"] = 50
    params["gamma"] = 0.99
    params["learning_rate"] = 0.001
    params["epsilon"] = 0.95
    params["l2_reg"] = 1e-3
    params["batch_size"] = 2048
    params["target_update"] = 10
    params["use_gpu"] = True
    params["gpuid"] = int(sys.argv[5]) if len(sys.argv) >= 6 else 0

    random_seed = 1234
    params["use_physics"] = False
    params[
        "full_attempt_limit"
    ] = (
        False
    )  # run to the full attempt limit, regardless of whether or not all solutions were found
    params["num_training_iters"] = 200
    params["num_training_trials"] = params["train_num_trials"]
    params["train_attempt_limit"] = 700

    params["num_testing_iters"] = 200
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
    agent = DQNAgent(env, 1, 1, params, require_log=False)
    trial_selected = agent.setup_trial(
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
    )
    env.reset()

    state_size = agent.env.observation_space.multi_discrete.shape[0]
    action_size = len(env.action_space)
    agent = DQNAgent(env, state_size, action_size, params)
    load_path = (
        sys.argv[3] if len(sys.argv) >= 4 and sys.argv[3] != "-" else ""
    )  # path without '.*' suffix
    transfer_tag = (
        sys.argv[4] if len(sys.argv) >= 5 and sys.argv[4] != "-" else ""
    )  # i.e. CC3toCC4
    save_path = os.path.join(
        params["data_dir"],
        "3rd_model_log/dqn-PE-{}-{}-{}".format(
            transfer_tag if transfer_tag else params["train_scenario_name"],
            params["reward_mode"],
            agent.subject_id,
        ),
    )
    os.makedirs(save_path, exist_ok=True)

    agent.env.reset()
    if load_path:
        agent.load(load_path)
        print("load model from {}".format(load_path))

    agent.env.human_agent = False
    agent.type_tag = "{}-{}-DQN-PE".format(
        transfer_tag if transfer_tag else params["train_scenario_name"],
        params["reward_mode"],
    )
    # train over multiple iterations over all trials
    fig = create_reward_fig()
    update_count = 0
    for iter_num in range(params["num_training_iters"]):
        agent.env.completed_trials = []
        for trial_num in range(0, params["num_training_trials"]):
            agent.run_trial_dqn(
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
            if len(agent.memory) > agent.batch_size:
                batch = agent.memory.sample(agent.batch_size)
                print("update with bs:{}".format(len(batch.state)))
                agent.update(batch, iter_num)
                update_count += 1
                if (update_count + 1) % params["target_update"]:
                    agent.target_q_net.load_state_dict(agent.q_net.state_dict())
        agent.save(save_path, iter_num)
    print(
        "Trial complete for subject {}. Average reward: {}".format(
            agent.logger.subject_id, agent.average_trial_rewards[-1]
        )
    )
    fig.savefig(os.path.join(save_path, "log.png"))


if __name__ == "__main__":
    main()
