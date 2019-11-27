#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : run_baselines.py
# Creation Date : 17-08-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

from subprocess import Popen, STDOUT
from random import shuffle
import itertools
import datetime
import torch
import os


num_device = 4
assert num_device <= torch.cuda.device_count()
base_path = os.path.dirname(os.path.abspath(__file__))
repeat = 5
agents = {
    "ppo": os.path.join(base_path, "PPO/ppo_open_lock.py"),
    "trpo": os.path.join(base_path, "TRPO/trpo_open_lock.py"),
    "a2c": os.path.join(base_path, "A2C/a2c_open_lock.py"),
    "dqn": os.path.join(base_path, "DQN/dqn_open_lock.py"),
    "dqn-PE": os.path.join(base_path, "DQN/dqn_open_lock_PE.py"),
}

scenerios = {
    "CC4": "0",  # 0/6 CC4
    "CE3": "1",  # 1/2 CE3
    "CC3": "3",  # 3/4 CC3
    "CE4": "5",  # CE4
}


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

FNULL = open(os.devnull, "w")
# import pickle
# args = pickle.load(open('./task_list.pkl', 'rb'))
# shuffle(args)

tmp = list(
    itertools.product(
        list(agents.values()), (scenerios.values()), rewards, range(repeat)
    )
)
shuffle(tmp)
args = []
for i in tmp:
    args.append(["python3"] + list(i[:-1]))

max_process_per_card = 10
max_process = max_process_per_card * num_device
ind = max_process

running_proc = []
for i, arg in enumerate(args[:ind]):
    print("[{}/{}] Start: {}".format(i + 1, len(args), arg))
    running_proc.append(
        (
            Popen(
                arg + ["-", "-", "{}".format(i % num_device)],
                stdout=FNULL,
                stderr=STDOUT,
            ),
            datetime.datetime.now(),
        )
    )

while running_proc:
    for (proc, start_time) in running_proc:
        ret = proc.poll()
        if ret is not None:
            now = datetime.datetime.now()
            print("End: {} @ {}, run for {}".format(proc.args, now, now - start_time))
            running_proc.remove((proc, start_time))
            if ind < len(args):
                print("[{}/{}] Start: {}".format(ind + 1, len(args), args[ind]))
                running_proc.append(
                    (
                        Popen(
                            args[ind] + ["-", "-", proc.args[-1]],
                            stdout=FNULL,
                            stderr=STDOUT,
                        ),
                        now,
                    )
                )
                ind += 1
        else:
            continue

if __name__ == "__main__":
    pass
