
import os
import json
import jsonpickle

import openlockagents.OpenLockLearner.util.plotter as plotter
from openlockagents.agent import load_agent

base_path = os.path.expanduser("~/Dropbox/Research/Causality/OpenLockExperiments/information_gain_vs_random_intervention_selection_tests/simplified_testing_scenario_2018_11_07")

random_subject_paths = [
    base_path + "/709583574422216339",
    base_path + "/965200786426647273",
    base_path + "/1753317527076959000",
]

ig_subject_paths = [
    base_path + "/1815854999685791507",
    base_path + "/2038030565531423632",
    base_path + "/2161172569307455466",
]


def main():
    data_dir = os.path.expanduser("~/Desktop")

    random_num_chains_with_belief_above_threshold_per_attempt = []
    information_gain_num_chains_with_belief_above_threshold_per_attempt = []

    for subject_path in random_subject_paths:
        agent = load_agent(subject_path)
        random_num_chains_with_belief_above_threshold_per_attempt.append(agent['num_chains_with_belief_above_threshold_per_attempt'])

    for subject_path in ig_subject_paths:
        agent = load_agent(subject_path)
        information_gain_num_chains_with_belief_above_threshold_per_attempt.append(agent['num_chains_with_belief_above_threshold_per_attempt'])


    # make data have same dimensionality everywhere and convert to numpy array
    data_random = plotter.pad_uneven_python_list(
        random_num_chains_with_belief_above_threshold_per_attempt
    )
    data_ig = plotter.pad_uneven_python_list(
        information_gain_num_chains_with_belief_above_threshold_per_attempt
    )

    models = [
        plotter.MultiRunPlotData("Random Intervention Selection", data_random),
        plotter.MultiRunPlotData(
            "Information gain-based Intervention Selection", data_ig
        ),
    ]

    plotter.create_plot_from_multi_run_plot_data(
        models,
        "Attempt number",
        "Number of chains with positive belief",
        "Information gain vs. random intervention selection",
        data_dir=data_dir,
        save_fig=False,
        show_fig=True,
    )

if __name__ == '__main__':
    main()