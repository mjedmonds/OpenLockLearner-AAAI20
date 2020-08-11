import time
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# must include this to unpickle properly

from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)
from openlockagents.common.agent import Agent

from openlockagents.OpenLockLearner.io.causal_structure_io import (
    load_causal_structures_from_file,
)
from openlockagents.OpenLockLearner.util.common import (
    parse_arguments,
    AblationParams,
    setup_structure_space_paths
)

from openlockagents.OpenLockLearner.main.generate_causal_structures import generate_causal_structures

from openlock.settings_trial import PARAMS
from openlock.common import generate_effect_probabilities


def plot_num_pruned(num_chains_pruned, filename):
    y = [math.log(x) if x > 0 else -0 for x in num_chains_pruned]
    if len(y) < 2:
        return None
    sns.set_style("dark")
    plt.plot(y)
    plt.ylabel("Num chains pruned (log)")
    plt.xlabel("Step number")
    fig = plt.gcf()
    fig.savefig(filename)
    plt.draw()
    return fig


def main():

    global_start_time = time.time()

    args = parse_arguments()

    ablation_params = AblationParams()

    if args.savedir is None:
        data_dir = "~/Desktop/Mass/OpenLockLearningResults/cc3-ce4_subjects"
    else:
        data_dir = args.savedir
    if args.scenario is None:
        param_scenario = "CC3-CE4"
    else:
        param_scenario = args.scenario
    if args.bypass_confirmation is None:
        bypass_confirmation = False
    else:
        bypass_confirmation = True
    if args.ablations is None:
        # ablation_params.INDEXED_DISTRIBUTIONS = True
        # ablation_params.PRUNING = True
        # ablation_params.TOP_DOWN_FIRST_TRIAL = True
        pass
    else:
        # process ablations
        for ablation in args.ablations:
            ablation = ablation.upper()
            if hasattr(ablation_params, ablation):
                setattr(ablation_params, ablation, True)
            else:
                exception_str = "Unknown ablation argument: {}".format(ablation)
                raise ValueError(exception_str)

    params = PARAMS[param_scenario]
    params["data_dir"] = data_dir
    params["train_attempt_limit"] = 30
    params["test_attempt_limit"] = 30
    # params['full_attempt_limit'] = True      # run to the full attempt limit, regardless of whether or not all solutions were found
    # run to the full attempt limit, regardless of whether or not all solutions were found
    params["full_attempt_limit"] = False
    params["intervention_sample_size"] = 10
    params["chain_sample_size"] = 1000
    params["use_physics"] = False

    # openlock learner params
    params["lambda_multiplier"] = 1
    params["local_alpha_update"] = 1
    params["global_alpha_update"] = 1
    params["epsilon"] = 0.99
    params["epsilon_decay"] = 0.99
    params["epsilon_active"] = False
    # these params were extracted using matlab
    # params["epsilon_ratios"] = [0.5422, 0.3079, 0.1287, 0.1067, 0, 0]
    params["intervention_mode"] = "action"
    # params["intervention_mode"] = 'attempt'
    # setup ablations
    params["ablation_params"] = ablation_params
    params["effect_probabilities"] = generate_effect_probabilities(l0=1, l1=1, l2=1, door=1)

    params["using_ids"] = False
    params["multiproc"] = False
    params["deterministic"] = False
    params["num_agent_runs"] = 40
    params["src_dir"] = "/tmp/openlocklearner/" + str(hash(time.time())) + "/src/"
    params["print_messages"] = False

    env = Agent.pre_instantiation_setup(params, bypass_confirmation)
    env.lever_index_mode = "position"

    causal_chain_structure_space_path, two_solution_schemas_structure_space_path, three_solution_schemas_structure_space_path = setup_structure_space_paths()

    if not os.path.exists(causal_chain_structure_space_path):
        print("WARNING: no hypothesis space files found, generating hypothesis spaces")
        generate_causal_structures()

    interventions_predefined = []
    # interventions_predefined = [("push_LOWERLEFT", "push_UPPERRIGHT", "push_door")]

    # these are used to advance to the next trial after there have no chains pruned for num_steps_with_no_pruning_to_finish_trial steps
    num_steps_with_no_pruning_to_finish_trial = 500
    num_agent_runs = params["num_agent_runs"]
    for i in range(num_agent_runs):
        agent_start_time = time.time()

        env = Agent.make_env(params)
        env.lever_index_mode = "position"

        causal_chain_structure_space, two_solution_schemas, three_solution_schemas = load_causal_structures_from_file(
            causal_chain_structure_space_path,
            two_solution_schemas_structure_space_path,
            three_solution_schemas_structure_space_path,
        )

        # setup agent
        agent = OpenLockLearnerAgent(
            env,
            causal_chain_structure_space,
            params,
            **{
                "two_solution_schemas": two_solution_schemas,
                "three_solution_schemas": three_solution_schemas,
            }
        )

        possible_trials = agent.get_random_order_of_possible_trials(
            params["train_scenario_name"]
        )

        agent.training_trial_order = possible_trials
        # training
        for trial_name in possible_trials:
            trial_selected, chain_idxs_pruned_from_initial_observation = agent.setup_trial(
                scenario_name=params["train_scenario_name"],
                action_limit=params["train_action_limit"],
                attempt_limit=params["train_attempt_limit"],
                specified_trial=trial_name,
            )

            agent.run_trial_openlock_learner(
                trial_selected,
                num_steps_with_no_pruning_to_finish_trial,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
                intervention_mode=params["intervention_mode"],
            )

        # testing
        if params["test_scenario_name"] == "CE4" or params["test_scenario_name"] == "CC4":
            trial_selected, chain_idxs_pruned_from_initial_observation = agent.setup_trial(
                scenario_name=params["test_scenario_name"],
                action_limit=params["test_action_limit"],
                attempt_limit=params["test_attempt_limit"],
            )

            agent.run_trial_openlock_learner(
                trial_selected,
                num_steps_with_no_pruning_to_finish_trial,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
                intervention_mode=params["intervention_mode"],
            )

        agent.print_agent_summary()
        print(
            "Finished agent. Total runtime: {}s".format(time.time() - agent_start_time)
        )
        agent.finish_subject("OpenLockLearner", "OpenLockLearner")

    print(
        "Finished all agents for {}. Total runtime: {}s".format(
            param_scenario, time.time() - global_start_time
        )
    )
    return


if __name__ == "__main__":
    main()
