import time
import itertools
import gym

from openlockagents.OpenLockLearner.util.common import (
    CAUSAL_CHAIN_EDGES,
    FIXED_STRUCTURE_ATTRIBUTES_GRAPH_PATH,
    HUMAN_PICKLE_DATA_PATH,
    THREE_LEVER_TRIALS,
    merge_perceptually_causal_relations_from_dict_of_trials,
    merge_solutions_from_dict_of_trials,
)
from openlockagents.OpenLockLearner.io.causal_structure_io import load_causal_chain_space
from openlockagents.common.io.log_io import load_solutions_by_trial
from openlockagents.OpenLockLearner.generator.chain_generator import (
    generate_causal_chains_fixed_structure_attributes,
    generate_true_chains,
)
from openlockagents.OpenLockLearner.perceptual_causality_python.perceptual_causality import (
    generate_perceptually_causal_relations,
    remove_state_from_perceptually_causal_relations,
)
from openlockagents.OpenLockLearner.learner.ChainPruner import prune_random_subsample

from openlock.settings_trial import PARAMS

def main():

    global_start_time = time.time()

    params = PARAMS["CE3-CE4"]

    trial_idx = 0
    trial_selected = THREE_LEVER_TRIALS[trial_idx]
    true_solutions = load_solutions_by_trial(HUMAN_PICKLE_DATA_PATH)
    true_chains_by_trial = generate_true_chains(true_solutions)
    true_chains_all_trials_merged = merge_solutions_from_dict_of_trials(
        true_chains_by_trial
    )

    data_dir = FIXED_STRUCTURE_ATTRIBUTES_GRAPH_PATH

    # causal_chain_space = load_chains(data_dir, 'uniform')

    chain_mode = "full"

    prune = True
    generate = True

    if generate:
        start_time = time.time()
        print("Generating perceptually causal relationships...")
        # causal_relations = None
        perceptually_causal_relations = generate_perceptually_causal_relations()
        # perceptually_causal_relations = load_perceptually_causal_relations()
        # remove door from state
        perceptually_causal_relations = remove_state_from_perceptually_causal_relations(
            perceptually_causal_relations, "door_lock"
        )
        perceptually_causal_relations_all_trials_merged = merge_perceptually_causal_relations_from_dict_of_trials(
            perceptually_causal_relations
        )
        # todo: changing this to a set removes the total number of info gains (frequency) we saw this relation
        perceptually_causal_relations_all_trials_merged = set(
            perceptually_causal_relations_all_trials_merged
        )
        print(
            "{} perceptually causal relationships found in human data. Took {}s".format(
                len(perceptually_causal_relations_all_trials_merged),
                time.time() - start_time,
            )
        )

        assert set(perceptually_causal_relations.keys()) == set(
            true_chains_by_trial.keys()
        ), "Perceptual relations and true chains do not have the same trials"


        # setup initial env
        env = gym.make("openlock-v1")
        env.use_physics = False
        env.initialize_for_scenario(params["train_scenario_name"])


        attributes_to_product = [
            env.attribute_labels[attribute] for attribute in env.attribute_order
        ]
        attribute_pairs = list(itertools.product(*attributes_to_product))


        # setup causal chain space
        start_time = time.time()

        state_space = list(attribute_pairs)
        action_space = env.actions

        # load chains
        # causal_chain_space = load_chains(data_dir, chain_mode)
        # generate chains
        causal_chain_space = generate_causal_chains_fixed_structure_attributes(
            state_space,
            action_space,
            attribute_order=env.attribute_order,
            lever_index_mode="position",
            structure=CAUSAL_CHAIN_EDGES,
            true_chains=true_chains_all_trials_merged,
            perceptually_causal_relations=perceptually_causal_relations_all_trials_merged,
            multiproc=True,
            data_dir=data_dir,
        )
        # return
        print("Load/generation time: {}s".format(time.time() - start_time))

        # start_time = time.time()
        # print('Checking for duplicate chains...')
        # # check for duplicate chains - no two chains in space should be identical
        # assert not causal_chain_space.check_for_duplicate_chains(), 'Duplicate chain found'
        # print('Checking for duplicate chains too {}s'.format(time.time()-start_time))

        # check for true chains
        # print(causal_chain_space.true_chains)
        # true_chain_idxs = causal_chain_space.verify_true_chains_in_causal_chains_with_positive_belief(causal_chain_space.true_chains)

    # must reload space
    print("RELOADING CHAINS...")
    start_time = time.time()
    causal_chain_space = load_causal_chain_space(data_dir, chain_mode)
    print("Load/generation time: {}s".format(time.time() - start_time))

    if prune:
        t = time.time()
        print("PRUNING RANDOM SUBSAMPLE OF CHAINS")
        causal_chain_space = prune_random_subsample(
            causal_chain_space, subset_size=1000
        )
        print("Pruning random subsample took {}s".format(time.time() - t))

    assert causal_chain_space.verify_idxs_have_belief_above_threshold(threshold=0.0)

    print("Finished. Total runtime: {}s".format(time.time() - global_start_time))

if __name__ == "__main__":
    main()
