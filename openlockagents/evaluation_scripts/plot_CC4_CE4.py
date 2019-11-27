#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import os
import glob
import pandas as pd
from collections import OrderedDict

def smooth(y, box_pts):
    box_pts = max(box_pts, 1) if len(y) > box_pts else 1
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

base_path = './3rd_model_log'

def process_file_list(fdict, max_l=1000):
    data = OrderedDict()
    for key, flist in fdict.items():
        dlist = []
        if key == 'MAML (Meta)':
            for f in flist:
                dlist.append(np.array(pd.read_pickle(f)[1][1][1][0])[:max_l])
        else:
            for f in flist:
                dlist.append(np.array(pd.read_pickle(f)[1][1][0])[:max_l])
        data[key] = np.array(dlist)
    return data

CC4_baseline_data_list = process_file_list(OrderedDict({
    'DQN' : glob.glob(os.path.join(base_path, 'all_log', 'dqn-CC4*')),
    'DQN-PE' : glob.glob(os.path.join(base_path, 'all_log', 'dqn-PE-CC4*')),
    'A2C' : glob.glob(os.path.join(base_path, 'all_log', 'a2c-CC4*')),
    'TRPO' : glob.glob(os.path.join(base_path, 'all_log', 'trpo-CC4*')),
    'PPO' : glob.glob(os.path.join(base_path, 'all_log', 'ppo-CC4*')),
    'MAML (Meta)': glob.glob(os.path.join(base_path, 'all_maml_log', 'maml-CC3*' )),
}))

CC4_transfer_data_list = process_file_list(OrderedDict({
    'DQN' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'dqn-CC3*')),
    'DQN-PE' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'dqn-PE-CC3*')),
    'A2C' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'a2c-CC3*')),
    'TRPO' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'trpo-CC3*')),
    'PPO' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'ppo-CC3*')),
    'MAML (Adapt)': glob.glob(os.path.join(base_path, 'all_k_shot_log', 'k_shot-CC4*' )),
}))

CE4_baseline_data_list = process_file_list(OrderedDict({
    'DQN' : glob.glob(os.path.join(base_path, 'all_log', 'dqn-CE4*')),
    'DQN-PE' : glob.glob(os.path.join(base_path, 'all_log', 'dqn-PE-CE4*')),
    'A2C' : glob.glob(os.path.join(base_path, 'all_log', 'a2c-CE4*')),
    'TRPO' : glob.glob(os.path.join(base_path, 'all_log', 'trpo-CE4*')),
    'PPO' : glob.glob(os.path.join(base_path, 'all_log', 'ppo-CE4*')),
    'MAML (Meta)': glob.glob(os.path.join(base_path, 'all_maml_log', 'maml-CE3*' )),
}))

CE4_transfer_data_list = process_file_list(OrderedDict({
    'DQN' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'dqn-CE3*')),
    'DQN-PE' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'dqn-PE-CE3*')),
    'A2C' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'a2c-CE3*')),
    'TRPO' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'trpo-CE3*')),
    'PPO' : glob.glob(os.path.join(base_path, 'all_transfer_log', 'ppo-CE3*')),
    'MAML (Adapt)': glob.glob(os.path.join(base_path, 'all_k_shot_log', 'k_shot-CE4*' )),
}))


tag_list = ['CC4', 'CE4']

value_list_CC4 = list(CC4_baseline_data_list.items()) + list(CC4_transfer_data_list.items())
value_list_CE4 = list(CE4_baseline_data_list.items()) + list(CE4_transfer_data_list.items())

fontsize = 35
ticksize = 25
title = 'Demo'
win_size = 50  # critic
a_min = np.iinfo(np.int64).min
a_max = np.iinfo(np.int64).max

def main():
    fig = plt.figure()
    fig.set_size_inches(17, 7)
    fig.clf()
    num_plots = len(tag_list) 
    flatui = ["#fc9272", "#de2d26", 
              "#bcbddc", "#756bb1", 
              "#7fcdbb", "#2c7fb8", 
              "#a1d99b", "#31a354", 
              "#bdbdbd", "#636363", 
              "#fa9fb5", "#dd1c77"]
    clrs = sns.color_palette(flatui)
    clrs = clrs[::2] + clrs[1::2]

    with sns.axes_style('darkgrid'):
        axes = fig.subplots(1, num_plots)
        for ind_model, item in enumerate(zip(value_list_CC4, value_list_CE4)):
            cc4, ce4 = item
            k, v_cc4 = cc4
            _, v_ce4 = ce4
            values_list = [v_cc4, v_ce4]
            for ind, (tag, values) in enumerate(zip(np.array(tag_list), np.array(values_list))):
                
                tmp = []
                for i in values:
                    tmp.append(smooth(i, win_size))
                tmp = np.array(tmp)
                mean = np.mean(tmp, axis=0)
                std = np.std(tmp, axis=0)

                axes[ind].plot(np.arange(len(mean)), mean, '-', c=clrs[ind_model], label=k)
                axes[ind].fill_between(np.arange(len(mean)), np.clip(mean - std, a_min=a_min, a_max=a_max), 
                                       np.clip(mean + std, a_min=a_min, a_max=a_max), alpha=0.3, facecolor=clrs[ind_model])
                axes[ind].set_title(tag, fontsize=fontsize)
                axes[ind].set_xlim(-50, 1000)
                axes[ind].set_ylim(0, 750)
                axes[ind].tick_params(labelsize=ticksize)

        axes[0].set_ylabel('Number of Attempts', fontsize=fontsize)
        axes[0].set_xlabel('Trials', fontsize=fontsize)
        axes[1].set_xlabel('Trials', fontsize=fontsize)

    fig.tight_layout(pad=0)
    fig.savefig('{}.pdf'.format(title))

if __name__ == '__main__':
    main()
