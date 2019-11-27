#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : eval_baselines.py
# Creation Date : 23-08-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import os
import glob
import json
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shutil import copyfile
from collections import OrderedDict

from openlockagents.common import ROOT_DIR


repeat = 5
agents = [
    # 'dqn',
    # 'dqn-PE',
    "a2c",
    "trpo",
    "ppo",
]

scenerios = ["CC3", "CE3", "CC4", "CE4"]

rewards = [
    "basic",
    # 'change_state',
    "unique_solutions",
    # 'change_state_unique_solutions',
    "negative_immovable",
    "negative_immovable_unique_solutions",
    # 'negative_immovable_partial_action_seq',
    # 'negative_immovable_negative_repeat',
    "negative_immovable_solution_multiplier",
    "negative_immovable_partial_action_seq_solution_multiplier",
    # 'negative_immovable_partial_action_seq_solution_multiplier_door_seq'
]

base_path = os.path.dirname(ROOT_DIR) + "/OpenLockRLResults/subjects"
log_path = os.path.join(base_path, "3rd_model_log")
target_path = os.path.join(log_path, "all_log")
target_img_path = os.path.join(log_path, "all_log", "imgs")
os.makedirs(target_path, exist_ok=True)
os.makedirs(target_img_path, exist_ok=True)

fontsize = 13


def extract_tag_value(subject_log_path):
    # agent.trial_length: 'Attempt Count Per Trial'
    # agent.trial_percent_attempt_success: 'Percentage of Successful Attempts in Trial'
    # agent.trial_percent_solution_found: 'Percentage of Solutions Found in Trial'
    # agent.average_trial_rewards: 'Average Trial Reward'
    # agent.attempt_rewards: 'Attempt Reward'

    trial_fl = glob.glob(os.path.join(subject_log_path, "*", "*.json"))
    trial_fl.sort()
    print("Extract from {} trials".format(len(trial_fl)))
    if len(trial_fl) not in [1000, 1200]:
        return None, None
    tag_list = [
        "Attempt Count Per Trial",
        "Percentage of Successful Attempts in Trial",
        "Percentage of Solutions Found in Trial",
        "Average Trial Reward",
        "Attempt Reward",
    ]
    attempt_count = []
    percentage_successful_attempt = []
    percentage_found_solution = []
    average_trial_reward = []
    attempt_reward = []
    for f in trial_fl:
        trial = json.load(open(f, "r"))
        num_attempt = len(trial["attempt_seq"])
        attempt_count.append(num_attempt)
        percentage_found_solution.append(
            len(trial["completed_solutions"]) / len(trial["solutions"])
        )
        average_trial_reward.append(trial["trial_reward"] / num_attempt)
        success_attempt_count = 0
        for attempt in trial["attempt_seq"]:
            attempt_reward.append(attempt["reward"])
            if attempt["success"]:
                success_attempt_count += 1
        percentage_successful_attempt.append(success_attempt_count / num_attempt)
    return (
        tag_list,
        [
            attempt_count,
            percentage_successful_attempt,
            percentage_found_solution,
            average_trial_reward,
            attempt_reward,
        ],
    )


def main():
    # copy log files
    # cart_res = list(itertools.product(agents, scenerios, rewards))

    # for i in cart_res:
    #     pattern = '-'.join(i) + '-'
    #     fl = glob.glob('{}/{}*'.format(log_path, pattern))
    #     print('Find {} results for {}'.format(len(fl), pattern))
    #     # assert(len(fl) == repeat)
    #     for ind, f in enumerate(fl):
    #         fname = os.path.join(f, 'log.pkl')
    #         copyfile(fname, os.path.join(target_path, '{}_{}_log.pkl'.format(pattern, ind)))

    # draw curves
    tag_list = []
    cart_res = list(itertools.product(scenerios, rewards))
    for plot_name in cart_res:  # each figure(casual+reward)
        agent_value_d = OrderedDict({})
        for a in agents:  # each curve(agent)
            pattern = a + "-" + "-".join(plot_name) + "-*"
            fl = glob.glob(os.path.join(log_path, pattern))
            all_value_list = []
            for f in fl:  # each run
                run_id = f.split("/")[-1].split("-")[-1]
                tag_list, value_list = extract_tag_value(
                    os.path.join(base_path, run_id)
                )
                if tag_list is None:
                    continue
                # remove last plot(attempt reward)
                tag_list = tag_list[:-1]
                value_list = value_list[:-1]
                if len(value_list[0]) not in [1200, 1000]:
                    continue
                all_value_list.append(value_list)
            agent_value_d[a] = all_value_list

        # average all the result
        tmp = OrderedDict({})
        for (k, v) in agent_value_d.items():  # each curve(agent)
            avg_value_list = [[] for i in tag_list]
            for v_l in v:  # each run
                for ind, v_t in enumerate(v_l):  # each tag
                    avg_value_list[ind].append(v_t)
            for ind, v_t in enumerate(avg_value_list):  # each tag
                # if np.array(v_t).shape[0] == 6:
                #     from IPython import embed; embed()

                print(np.array(v_t).shape)
                v_t = np.array(v_t).mean(0)
                avg_value_list[ind] = v_t
            tmp[k] = avg_value_list
        agent_value_d = tmp

        fig = plt.figure()
        fig.set_size_inches(20, 5)
        window_size = 100
        # def log_values(self, values_list, fig, taglist, title):
        def smooth(y, box_pts):
            box_pts = max(box_pts, 1) if len(y) > box_pts else 1
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode="valid")
            return y_smooth

        fig.clf()
        num_plots = len(tag_list)
        clrs = sns.color_palette("husl", len(agent_value_d))
        base = fig.add_subplot(111)
        base.spines["top"].set_color("none")
        base.spines["bottom"].set_color("none")
        base.spines["left"].set_color("none")
        base.spines["right"].set_color("none")
        base.tick_params(
            labelcolor="w", top="off", bottom="off", left="off", right="off"
        )
        with sns.axes_style("darkgrid"):
            axes = fig.subplots(1, num_plots)
            ratio = []
            for ind_agent, (k, v) in enumerate(agent_value_d.items()):
                values_list = v
                ratio.append(
                    [
                        k,
                        (
                            np.array(values_list[2][-10:])
                            / np.array(values_list[0][-10:])
                        ).mean(),
                    ]
                )
                for ind, (tag, values) in enumerate(zip(tag_list, values_list)):
                    axes[ind].plot(
                        np.arange(len(values)),
                        values,
                        "-",
                        c=clrs[ind_agent],
                        alpha=0.3,
                    )
                    res = smooth(values, window_size)
                    axes[ind].plot(
                        np.arange(len(res)),
                        res,
                        "-",
                        c=clrs[ind_agent],
                        label=k.upper(),
                    )
                    # axes[ind].fill_between(epochs, mins, maxs, alpha=0.3, facecolor=clrs[0])
                    axes[ind].set_ylabel(tag, fontsize=fontsize)

            axes[-1].legend(fontsize=fontsize, loc=4)

        base.set_title("-".join(plot_name), fontsize=fontsize)
        base.set_xlabel("Trials", fontsize=fontsize)
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(target_img_path, "{}.pdf".format("-".join(plot_name))))
        np.savetxt(
            os.path.join(target_img_path, "{}.txt".format("-".join(plot_name))),
            np.array(ratio),
            fmt="%s",
            delimiter=",",
        )


if __name__ == "__main__":
    main()
