import gym
import numpy as np
import time
import os

from openlockagents.OpenLockLearner.util.common import (
    FIXED_STRUCTURE_ATTRIBUTES_GRAPH_TWO_STEP_TESTING_PATH,
)
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)
from openlockagents.OpenLockLearner.main.two_step_testing_scenario import (
    TWO_STEP_STRUCTURE,
    generate_perceptually_causal_relations_two_step_testing_scenario,
)
from openlockagents.OpenLockLearner.util.setup_util import setup_causal_chain_space, create_and_run_agent
import openlockagents.OpenLockLearner.util.plotter as plotter
from openlockagents.OpenLockLearner.experiments.IG_vs_random_intervention_common import run_experiment
from openlock.settings_trial import PARAMS


def main():
    fake_model_data = [
        plotter.MultiRunPlotData("Fake 1", np.random.rand(5, 1000)),
        plotter.MultiRunPlotData("Fake 2", np.random.rand(5, 1000)),
    ]
    # plotter.create_plot_from_multi_run_plot_data(fake_model_data, "Fake x-axis", "Fake y-axis", "Fake test plot", data_dir=os.path.expanduser("~/Desktop"))

    # compares if random intervention selection is better than intervention selection based on information gain

    global_start_time = time.time()

    params = PARAMS["CE3-CE4"]
    params["train_scenario_name"] = "TwoStepTestingScenario"
    params["test_scenario_name"] = "TwoStepTestingScenario"
    params["data_dir"] = "~/Desktop/OpenLockLearningResultsTesting/subjects"
    params["train_attempt_limit"] = 10000
    params["test_attempt_limit"] = 10000
    # params['full_attempt_limit'] = True      # run to the full attempt limit, regardless of whether or not all solutions were found
    # run to the full attempt limit, regardless of whether or not all solutions were found
    params["full_attempt_limit"] = False
    # params["intervention_sample_size"] = None
    params["intervention_sample_size"] = 10
    # params["chain_sample_size"] = None
    params["chain_sample_size"] = 1000

    params["using_ids"] = True
    params["multiproc"] = True

    params['prune_chain_space'] = False
    params['generate_chains'] = True

    params["chain_data_dir"] = FIXED_STRUCTURE_ATTRIBUTES_GRAPH_TWO_STEP_TESTING_PATH
    params["chain_mode"] = "full"

    np.random.seed(1234)

    structure = TWO_STEP_STRUCTURE

    perceptually_causal_relations = (
        generate_perceptually_causal_relations_two_step_testing_scenario()
    )

    num_runs = 3

    run_experiment(
        params=params,
        structure=structure,
        perceptually_causal_relations=perceptually_causal_relations,
        num_runs=num_runs,
    )

    print("Finished. Total runtime: {}s".format(time.time() - global_start_time))
    return


if __name__ == "__main__":
    main()
