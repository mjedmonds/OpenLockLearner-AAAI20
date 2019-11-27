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
from openlockagents.MAML.maml_agent import MAMLAgent
from openlockagents.MAML.utils.replay_memory import Memory


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
    params_list = []
    env_list = ["CE3-CE4", "CE4"]
    memory_list = []
    fig_list = [create_reward_fig() for _ in env_list]
    log_list = [[] for _ in env_list]
    for env_name in env_list:
        # if len(sys.argv) < 2:
        #     # general params
        #     # training params
        #     # PICK ONE and comment others
        #     params = PARAMS['CE3-CE4']
        #     # params = PARAMS['CE3-CC4']
        #     # params = PARAMS['CC3-CE4']
        #     # params = PARAMS['CC3-CC4']
        #     # params = PARAMS['CE4']
        #     # params = PARAMS['CC4']
        # else:
        #     setting = sys.argv[1]
        #     params = PARAMS[IDX_TO_PARAMS[int(setting) - 1]]
        #     print('training_scenario: {}, testing_scenario: {}'.format(params['train_scenario_name'],
        #                                                                params['test_scenario_name']))
        #     params['reward_mode'] = sys.argv[2]
        memory_list.append(Memory())
        # generic
        params = PARAMS[env_name]
        params["gamma"] = 0.99
        params["reward_mode"] = "basic"
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
        params["num_grad_update"] = 1
        params["lr_pre_update"] = 1e-3
        params["lr_meta_update"] = 1e-3
        params["pre_batch_size"] = 2048
        params["meta_batch_size"] = 2048

        # others
        params["use_gpu"] = True and torch.cuda.is_available()
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
        params["num_training_iters"] = 1000
        params["num_training_trials"] = params["train_num_trials"]
        params["train_attempt_limit"] = 700

        params["num_testing_iters"] = 1000
        params["num_testing_trials"] = params["test_num_trials"]
        params["test_attempt_limit"] = 700

        # RL specific settings
        params["data_dir"] = os.path.dirname(ROOT_DIR) + "/OpenLockRLResults/subjects"

        params_list.append(params)

    # TODO: we assume all the scenarios share the same observation space
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
    agent = MAMLAgent(env, 1, 1, params, env_list, require_log=False)
    trial_selected = agent.setup_trial(
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
    )
    env.reset()

    state_size = agent.env.observation_space.multi_discrete.shape[0]
    action_size = len(env.action_space)
    agent = MAMLAgent(env, state_size, action_size, params, env_list)
    save_path = os.path.join(
        params["data_dir"],
        "3rd_model_log/maml-{}-{}-{}".format(
            "_".join(env_list), params["reward_mode"], agent.subject_id
        ),
    )
    load_path = sys.argv[3] if len(sys.argv) >= 4 else ""  # path without '.*' suffix
    os.makedirs(save_path, exist_ok=True)
    reward_counter_list = [env.reward_strategy.counter for _ in env_list]
    reward_attempt_count_list = [env.reward_strategy.attempt_count for _ in env_list]

    agent.env.reset()
    if load_path:
        agent.load(load_path)
        print("load model from {}".format(load_path))

    agent.env.human_agent = False
    agent.type_tag = "{}-{}-MAML".format("_".join(env_list), params["reward_mode"])
    # train over multiple iterations over all trials
    for iter_num in range(params_list[0]["num_training_iters"]):
        for ind, env_name in enumerate(env_list):
            agent.env.completed_trials = []
            agent.env.scenario = None
            agent.env.cur_trial = None
            agent.env.reward_strategy.counter = reward_counter_list[ind]
            agent.env.reward_strategy.attempt_count = reward_attempt_count_list[ind]
            print("[Train] Now meta train on {}".format(env_name))
            params = params_list[ind]
            memory = memory_list[ind]
            agent._update_params_and_mem(params, memory)

            for trial_num in range(0, params_list[0]["num_training_trials"]):
                agent.run_trial_maml(
                    scenario_name=params["train_scenario_name"],
                    fig=fig_list[ind],
                    action_limit=params["train_action_limit"],
                    attempt_limit=params["train_attempt_limit"],
                    trial_count=trial_num,
                    iter_num=iter_num,
                    env_ind=ind,
                )
                fig_list[ind], log_list[ind] = agent.log_values(
                    [
                        agent.trial_length[ind],
                        agent.trial_percent_attempt_success[ind],
                        agent.trial_percent_solution_found[ind],
                        agent.average_trial_rewards[ind],
                        agent.attempt_rewards[ind],
                    ],
                    fig_list[ind],
                    [
                        "Attempt Count Per Trial",
                        "Percentage of Successful Attempts in Trial",
                        "Percentage of Solutions Found in Trial",
                        "Average Trial Reward",
                        "Attempt Reward",
                    ],
                    agent.type_tag + "-{}".format(env_name),
                )
        pickle.dump(
            (agent.type_tag, log_list, params),
            open(os.path.join(save_path, "log.pkl"), "wb"),
        )
        # update
        for ind, env_name in enumerate(env_list):
            memory = memory_list[ind]
            params = params_list[ind]
            batch_a, batch_b = [], []
            if len(memory) > params["pre_batch_size"] + params["meta_batch_size"]:
                print("[Update] Now do an update with {}".format(env_name))
                batch_a.append(memory.sample(params["pre_batch_size"]))
                batch_b.append(memory.sample(params["meta_batch_size"]))
                memory.clear()
        print("[Update] Now update")
        agent.update(batch_a, batch_b, iter_num)
        agent.save(save_path, iter_num)
    print(
        "Trial complete for subject {}. Average reward: {}".format(
            agent.logger.subject_id, agent.average_trial_rewards[-1]
        )
    )
    fig.savefig(os.path.join(save_path, "log.png"))


if __name__ == "__main__":
    main()
