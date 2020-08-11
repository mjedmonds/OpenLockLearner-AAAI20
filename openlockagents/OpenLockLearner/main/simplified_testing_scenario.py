import numpy as np
import time
import os

from openlockagents.OpenLockLearner.util.common import CAUSAL_CHAIN_EDGES
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalObservation,
    CausalRelationType,
    CausalRelation,
)
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)


from openlockagents.common.agent import Agent

from openlockagents.OpenLockLearner.util.common import FLUENTS, FLUENT_STATES, ACTIONS, AblationParams

from openlockagents.OpenLockLearner.io.causal_structure_io import (
    load_causal_structures_from_file,
)

from openlockagents.OpenLockLearner.causal_classes.hypothesis_space import generate_hypothesis_space

from openlock.settings_trial import PARAMS
from openlock.common import generate_effect_probabilities


def generate_perceptually_causal_relations_simplified_testing_scenario():
    perceptually_causal_relations = [
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("UPPERLEFT", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("UPPERLEFT", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("UPPERRIGHT", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("UPPERRIGHT", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("LOWERLEFT", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("LOWERLEFT", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("LOWERRIGHT", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("LOWERRIGHT", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None,
            ),
            None,
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("door", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None,
            ),
            None,
        ),
    ]
    return perceptually_causal_relations


def main():

    global_start_time = time.time()

    param_scenario = "CE3-CE4"
    params = PARAMS[param_scenario]
    params["data_dir"] = "~/Desktop/OpenLockLearningResultsTesting/subjects"
    params["train_scenario_name"] = "CE3_simplified"
    params["test_scenario_name"] = "CE3_simplified"
    params["train_attempt_limit"] = 10000
    params["test_attempt_limit"] = 10000
    # params['full_attempt_limit'] = True      # run to the full attempt limit, regardless of whether or not all solutions were found
    # run to the full attempt limit, regardless of whether or not all solutions were found
    params["full_attempt_limit"] = False
    params["intervention_sample_size"] = 10
    params["chain_sample_size"] = 1000

    # openlock learner params
    params["lambda_multiplier"] = 1
    params["local_alpha_update"] = 2
    params["global_alpha_update"] = 1
    params["epsilon"] = 0.99
    params["epsilon_decay"] = 0.99
    params["epsilon_active"] = False
    params["intervention_mode"] = "action"
    # params["intervention_mode"] = 'attempt'
    # setup ablations
    ablation_params = AblationParams()
    # ablation_params.INDEXED_DISTRIBUTIONS = True
    # ablation_params.PRUNING = True
    # ablation_params.TOP_DOWN_FIRST_TRIAL = True
    params["ablation_params"] = ablation_params
    params["effect_probabilities"] = generate_effect_probabilities(l0=1, l1=1, l2=1, door=1)
    params["using_ids"] = False
    params["multiproc"] = False
    params["use_physics"] = False

    params["deterministic"] = False
    params["num_agent_runs"] = 40
    params["src_dir"] = "/tmp/openlocklearner/" + str(hash(time.time())) + "/src/"

    np.random.seed(1234)

    env = Agent.pre_instantiation_setup(params)
    env.lever_index_mode = "position"

    attributes = [env.attribute_labels[attribute] for attribute in env.attribute_order]

    structure = CAUSAL_CHAIN_EDGES


    generate_causal_structures = False
    causal_chain_structure_space_path = os.path.expanduser(
        "~/Desktop/simplified_causal_chain_space.pickle"
    )
    two_solution_schemas_structure_space_path = os.path.expanduser(
        "~/Desktop/simplified_two_solution_schemas.pickle"
    )
    three_solution_schemas_structure_space_path = os.path.expanduser(
        "~/Desktop/simplified_three_solution_schemas.pickle"
    )
    if generate_causal_structures:
        # perceptually_causal_relations = (
        #     generate_perceptually_causal_relations_simplified_testing_scenario()
        # )
        perceptually_causal_relations = None
        causal_chain_structure_space = generate_chain_structure_space(
            env=env,
            actions=ACTIONS,
            attributes=attributes,
            fluents=FLUENTS,
            fluent_states=FLUENT_STATES,
            perceptually_causal_relations=perceptually_causal_relations,
            structure=structure,
        )
        write_causal_structure_space(
            causal_chain_structure_space=causal_chain_structure_space,
            causal_chain_structure_space_path=causal_chain_structure_space_path,
        )

        t = time.time()

        two_solution_schemas = AbstractSchemaStructureSpace(
            causal_chain_structure_space.structure, 2, draw_chains=False
        )
        write_schema_structure_space(
            schema_structure_space=two_solution_schemas,
            schema_structure_space_path=two_solution_schemas_structure_space_path,
        )

        three_solution_schemas = AbstractSchemaStructureSpace(
            causal_chain_structure_space.structure, 3, draw_chains=False
        )
        write_schema_structure_space(
            schema_structure_space=three_solution_schemas,
            schema_structure_space_path=three_solution_schemas_structure_space_path,
        )

        print("Schema generation time: {}s".format(time.time() - t))

        return

    # these are used to advance to the next trial after there have no chains pruned for num_steps_with_no_pruning_to_finish_trial steps
    num_steps_with_no_pruning_to_finish_trial = 5

    interventions_predefined = []
    # interventions_predefined = [
    #     ("push_LEFT", "pull_LEFT"),   # pushing and pulling the same lever
    #     ("push_UPPER", "pull_UPPER"), # unlocking and locking the door
    #     ("pull_LEFT", "pull_UPPER"),  # no state change
    #     ("push_LEFT", "pull_UPPER"),  # left lever state change
    #     ("push_LEFT", "push_UPPER"),  # door unlocks at the last stage
    #     ("push_LEFT", "push_UPPER"),  # door unlocks at the last stage
    #     # ("push_UPPER", "push_door"),  # only solution
    # ]

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

        # training
        agent.training_trial_order = possible_trials
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


if __name__ == "__main__":
    main()
