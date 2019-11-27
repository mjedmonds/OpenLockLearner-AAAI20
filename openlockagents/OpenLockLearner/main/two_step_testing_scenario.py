import gym
import numpy as np
import time

from openlockagents.OpenLockLearner.util.common import (
    FIXED_STRUCTURE_ATTRIBUTES_GRAPH_TWO_STEP_TESTING_PATH,
    GRAPH_INT_TYPE,
)
from openlockagents.OpenLockLearner.util.util import generate_solutions_by_trial
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalObservation,
    CausalRelationType,
    CausalRelation
)
from openlockagents.OpenLockLearner.causal_classes.CausalChainManagerChains.CausalChainManagerChains import CausalChainManagerChains
from openlockagents.OpenLockLearner.causal_classes.CausalChainStructureSpace import CausalChainStructureSpace
from openlockagents.OpenLockLearner.util.common import FLUENTS, FLUENT_STATES, ACTIONS
from openlockagents.OpenLockLearner.causal_classes.CausalChainManagerChains.CausalRelationManager import CausalRelationManager
from openlockagents.OpenLockLearner.causal_classes.CausalChainManagerChains.CausalChainManager import CausalChainManager
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)
from openlock.settings_trial import PARAMS

TWO_STEP_STRUCTURE = (
    ("action0", "state0"),
    ("action1", "state1"),
    ("state0", "state1"),
)


def generate_perceptually_causal_relations_two_step_testing_scenario():
    perceptually_causal_relations = [
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("LEFT", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None
            ),
            None
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("LEFT", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None
            ),
            None
        ),
        CausalObservation(
            CausalRelation(
                action="pull",
                attributes=("UPPER", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None
            ),
            None
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("UPPER", "GREY"),
                causal_relation_type=CausalRelationType.one_to_zero,
                precondition=None
            ),
            None
        ),
        CausalObservation(
            CausalRelation(
                action="push",
                attributes=("door", "GREY"),
                causal_relation_type=CausalRelationType.zero_to_one,
                precondition=None
            ),
            None
        ),
    ]
    return perceptually_causal_relations

def main():
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

    params["using_ids"] = False
    params["multiproc"] = False

    params['prune_chain_space'] = False
    params['generate_chains'] = True

    structure = TWO_STEP_STRUCTURE

    perceptually_causal_relations = (
        generate_perceptually_causal_relations_two_step_testing_scenario()
    )

    # setup initial env
    env = gym.make("openlock-v1")
    env.use_physics = False
    env.lever_index_mode = "position"
    env.initialize_for_scenario(params["train_scenario_name"])

    attributes = [
        env.attribute_labels[attribute] for attribute in env.attribute_order
    ]

    causal_relation_manager = CausalRelationManager(ACTIONS, attributes, FLUENTS, FLUENT_STATES, perceptually_causal_relations)
    causal_chain_manager = CausalChainManager(causal_relation_manager, chain_length=2)
    print("Chain generation time: {}s".format(time.time() - global_start_time))

    causal_chain_space = CausalChainStructureSpace(
        attributes=attributes,
        structure=structure,
        attribute_order=env.attribute_order,
        lever_index_mode="position",
    )

    causal_chains = CausalChainManagerChains()
    causal_chains.set_chains(causal_chain_manager.chains)
    causal_chain_space.causal_chains = causal_chains

    # setup agent
    agent = OpenLockLearnerAgent(env, causal_chain_space, params)

    trial_selected = agent.setup_trial(
        params["train_scenario_name"],
        action_limit=2,
        attempt_limit=100,
    )

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

    # run the trial
    agent.run_trial_openlock_learner_attempt_intervention(
        trial_selected,
        num_steps_with_no_pruning_to_finish_trial,
        interventions_predefined=interventions_predefined,
    )

    print("Finished. Total runtime: {}s".format(time.time() - global_start_time))
    return


if __name__ == '__main__':
    main()
