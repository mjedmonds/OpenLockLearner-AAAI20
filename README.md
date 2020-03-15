# OpenLock Learner

This repo contains the code for the agents in the AAAI 2020 Oral paper "Theory-based Causal Transfer: Integrating Instance-level Induction and Abstract-level Structure Learning." For the paper and additional details, please see the [project page](https://mjedmonds.com/projects/OpenLock/AAAI20_OpenLockLearner.html).

It provides various agents to use with the [OpenLock OpenAI Gym environment](https://github.com/mjedmonds/OpenLock)

You'll need to bring in the OpenLock repo into your `PYTHONPATH` for this repo. If you are using a virtualenv, you can add `export PYTHONPATH="/path/to/OpenLock"` to `/bin/activate`. Alternatively, you can `python setup.py install` OpenLock to add it as a package to your python environment.

## Installation

1. Clone this repo, cd into it, and run `python setup.py install`
2. Follow OpenLock instructions here: [https://github.com/mjedmonds/OpenLock](https://github.com/mjedmonds/OpenLock)

## Execution

You can run any of the agents in the `openlockagents` directory. The main python file for each agent will end with `_open_lock.py`. To run the model presented in the paper, run:
```
python openlockagents/OpenLockLearner/main/openlock_learner_open_lock.py 
```

If you use this repo, please cite our work:

```
@inproceedings{edmonds2020theory,
  title={Theory-based Causal Transfer: Integrating Instance-level Induction and Abstract-level Structure Learning},
  author={Edmonds, Mark and Ma, Xiaojian and Qi, Siyuan, and Zhu, Yixin and Lu, Hongjing and Zhu, Song-Chun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```


