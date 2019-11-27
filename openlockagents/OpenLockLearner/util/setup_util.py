import itertools
import time

from openlockagents.OpenLockLearner.learner.ChainPruner import prune_random_subsample
from openlockagents.OpenLockLearner.generator.chain_generator import generate_causal_chains_fixed_structure_attributes
from openlockagents.OpenLockLearner.io.causal_structure_io import load_causal_chain_space
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import OpenLockLearnerAgent


def setup_causal_chain_space(
        env,
        structure,
        perceptually_causal_relations,
        multiproc,
        data_dir,
        chain_mode,
        prune_chain_space,
        generate,
        using_ids
):
    # generate chains
    if generate:
        attributes_to_product = [
            env.attribute_labels[attribute] for attribute in env.attribute_order
        ]
        attribute_pairs = list(itertools.product(*attributes_to_product))

        # setup perceptually causal relations
        # all_possible_perceptually_causal_relations = [
        #     CausalObservation(state=s, causal_relation_type=r, action=a, attributes=attr)
        #     for s, r, a, attr in itertools.product(
        #         states, causal_relations, actions, attribute_pairs
        #     )
        # ]

        # setup causal chain space
        state_space = list(attribute_pairs)
        action_space = env.actions
        causal_chain_space = generate_causal_chains_fixed_structure_attributes(
            state_space,
            action_space,
            attribute_order=env.attribute_order,
            lever_index_mode="position",
            structure=structure,
            perceptually_causal_relations=perceptually_causal_relations,
            multiproc=multiproc,
            data_dir=data_dir,
            batch_size=100000,
            using_ids=using_ids
        )

        # must reload space
        print("RELOADING CHAINS...")

    start_time = time.time()
    causal_chain_space = load_causal_chain_space(data_dir, chain_mode)
    print("Load/generation time: {}s".format(time.time() - start_time))

    if prune_chain_space:
        print("PRUNING RANDOM SUBSAMPLE OF CHAINS")

        t = time.time()
        causal_chain_space = prune_random_subsample(
            causal_chain_space, subset_size=1000
        )
        print("Pruning random subsample took {}s".format(time.time() - t))
    return causal_chain_space


def create_and_run_agent(
    env, causal_chain_space, params, use_random_intervention
):
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

    # run the trial without
    agent.run_trial_openlock_learner_attempt_intervention(
        trial_selected,
        num_steps_with_no_pruning_to_finish_trial,
        interventions_predefined=interventions_predefined,
        use_random_intervention=use_random_intervention,
    )
    return agent
