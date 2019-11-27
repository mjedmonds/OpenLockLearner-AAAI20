import re
import time
import zmq
import zlib
import pickle
import matplotlib.pyplot as plt
import atexit

# must include this to unpickle properly
from openlockagents.OpenLockLearner.causal_classes.CausalChain import (
    CausalChainCompact,
)
from openlockagents.OpenLockLearner.util.common import (
    STATES_ROLE,
    ACTIONS_ROLE,
    ACTION_REGEX_STR,
    STATE_REGEX_STR,
    THREE_LEVER_TRIALS,
)
from openlockagents.OpenLockLearner.io.causal_structure_io import load_causal_chains
from openlockagents.OpenLockLearner.perceptual_causality_python.perceptual_causality import (
    load_perceptually_causal_relations,
)
from openlockagents.OpenLockLearner.causal_classes.OutcomeSpace import OutcomeSpace
from openlockagents.OpenLockLearner.learner.OpenLockLearnerAgent import (
    OpenLockLearnerAgent,
)


def exit_handler(agent, fig):
    agent.finish_subject()


def send_zipped_pickle(socket, obj, flags=0, protocol=-1):
    """pickle an object, and zip the pickle before sending it"""
    p = pickle.dumps(obj, protocol)
    z = zlib.compress(p)
    return socket.send(z, flags=flags)


def recv_zipped_pickle(socket, flags=0, protocol=-1):
    """inverse of send_zipped_pickle"""
    z = socket.recv(flags)
    p = zlib.decompress(z)
    return pickle.loads(p)


def send_chain_actions(socket, chains):
    for true_chain in chains:
        action_seq_dict = {"action_sequence": true_chain.actions}
        send_zipped_pickle(socket, action_seq_dict)
        print("Sent {} to the simulator".format(action_seq_dict))
        result = recv_zipped_pickle(socket)


def main():

    params = dict()
    params["data_dir"] = "~/Desktop/OpenLockLearningResults/subjects"
    params["train_attempt_limit"] = 10000
    params["test_attempt_limit"] = 10000
    params[
        "full_attempt_limit"
    ] = (
        True
    )  # run to the full attempt limit, regardless of whether or not all solutions were found
    params["intervention_sample_size"] = 10
    params["chain_sample_size"] = 1000

    trial_idx = 0
    selected_trial = THREE_LEVER_TRIALS[trial_idx]

    # chain_mode = 'full'
    # chain_mode = 'pruned'
    chain_mode = "subsampled"

    prune_perceptual = False
    prune_subsample = False

    # load chains
    causal_chain_space = None
    if chain_mode == "full":
        causal_chain_space = load_causal_chains(mode="fixed", chain_dir="/chains")
        causal_chain_space.set_uniform_belief_for_causal_chains_with_positive_belief()
    elif chain_mode == "pruned":
        causal_chain_space = load_causal_chains(
            mode="fixed", chain_dir="/chains_pruned"
        )
    elif chain_mode == "subsampled":
        causal_chain_space = load_causal_chains(
            mode="fixed", chain_dir="/chains_subsampled"
        )
    print("Loaded {} chains from disk".format(len(causal_chain_space.causal_chains)))

    # OPTION 1: Prune/run perceptual causality
    # load causal relations
    if prune_perceptual:
        perceptual_causal_relations = load_perceptually_causal_relations()
        causal_chain_space.prune_space_from_constraints(
            perceptual_causal_relations[selected_trial]
        )
        causal_chain_space.set_uniform_prior_for_chains_with_positive_belief()
        causal_chain_space.write_chains_with_positive_belief()
        return

    # Option 2: prune random subset. Would recommend running after pruning based on perceptual causality
    # extract a random subset of chains and save them for testing/debugging purposes
    if prune_subsample:
        causal_chain_space.prune_space_random_subset(subset_size=1000)
        causal_chain_space.set_uniform_prior_for_chains_with_positive_belief()
        causal_chain_space.write_chains_with_positive_belief("/chains_subsampled")
        return

    # verify all causally plausible chains are present in chain space
    # todo: this does not work because causal chain space does not allow duplicate states. e.g. l0 cannot be used twice
    # plausible_chains = CAUSALLY_PLAUSIBLE_CHAINS_CE3
    # causal_chain_space.verify_chains_present_in_chain_space(plausible_chains)
    # return

    # Add in specific chains that are useful for testing
    causal_chain_space.add_chain(
        CausalChainCompact(
            ("l1", "l0", "door"), ("push_l1", "push_l0", "push_door"), (1, 0, 1)
        )
    )

    # sanity check to make sure we didn't accidentally prune out the true chains
    # true_chains_in_hypothesis_space = causal_chain_space.find_true_chain_indices_and_positive_beliefs()
    # assert true_chains_in_hypothesis_space

    # setup spaces
    # intervention_sample_size = None
    num_actions_in_chain = len(
        [
            x
            for x in causal_chain_space.base_schema.node_id_to_node_dict.keys()
            if re.match(ACTION_REGEX_STR, x)
        ]
    )
    intervention_space = InterventionSpace(
        ACTIONS_ROLE, num_actions_in_chain, params["intervention_sample_size"]
    )
    num_states_in_chain = len(
        [
            x
            for x in causal_chain_space.base_schema.node_id_to_node_dict.keys()
            if re.match(STATE_REGEX_STR, x)
        ]
    )
    outcome_space = OutcomeSpace(STATES_ROLE, num_states_in_chain)

    # setup agent
    agent = OpenLockLearnerAgent(
        None, causal_chain_space, intervention_space, outcome_space, params
    )

    fig = plt.figure()
    plt.ion()
    fig.show()

    # register exit handler
    atexit.register(exit_handler, agent, fig)

    # setup socket
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5555")

    # reset simulator
    send_zipped_pickle(socket, "reset")
    response_str = recv_zipped_pickle(socket)
    print(response_str)

    send_zipped_pickle(
        socket,
        {
            "attempt_limit": (
                params["train_attempt_limit"],
                params["test_attempt_limit"],
                params["full_attempt_limit"],
            )
        },
    )
    response_str = recv_zipped_pickle(socket)
    print(response_str)

    send_zipped_pickle(socket, "get_current_scenario_name")
    scenario_name = recv_zipped_pickle(socket)
    print("Received current scenario from simulator: {}".format(scenario_name))

    send_zipped_pickle(socket, {"set_trial": selected_trial})
    response_str = recv_zipped_pickle(socket)
    print(response_str)

    agent.true_chains = true_chain_selector(scenario_name)

    # get trial info from simulator
    send_zipped_pickle(socket, "get_current_trial")
    agent.logger.cur_trial = recv_zipped_pickle(socket)
    print("Received current trial info from simulator")

    optimal_interventions, optimal_outcomes = (
        agent.get_true_interventions_and_outcomes()
    )

    # interventions = [
    #     ('push_l1', 'push_l0', 'push_door'),
    #     ('push_l0', 'push_inactive2', 'push_door'),
    #     ('push_l1', 'push_l0', 'push_inactive0'),
    #     ('push_l1', 'pull_inactive0', 'push_door'),
    #     # ('push_l2', 'push_l0', 'push_door'),
    #     # ('push_l2', 'push_inactive2', 'push_door'),
    #     # ('push_l2', 'push_l0', 'push_inactive0'),
    #     # ('push_l2', 'pull_inactive0', 'push_door'),
    # ]

    # send_chain_actions(socket, agent.true_chains)

    t = time.time()
    chains_with_positive_belief = agent.causal_chain_structure_space.chains_with_positive_belief()

    trial_finished = False

    num_chains_pruned = []
    while True:
        num_chains_pruned_step = 0
        # sample from interventions
        # control sample size to always be smaller than valid chain space
        interventions = agent.sample_intervention_batch(agent.intervention_sample_size)
        outcomes = outcome_space.outcomes

        # intervention, intervention_info_gain = agent.causal_chain_space.select_intervention(optimal_interventions, optimal_outcomes)
        intervention, intervention_info_gain = agent.select_intervention(
            interventions, outcomes, sample_chains=True
        )
        # intervention = causal_chain_space.select_intervention_multiprocessing(process_pool, num_processors, intervention_space, outcome_space)
        print(
            "Step {}: Computed optimal intervention {} with info gain {:0.4f}. Took {:0.6f} seconds".format(
                agent.attempt_count,
                intervention,
                intervention_info_gain,
                time.time() - t,
            )
        )

        # send the intervention to the simulator
        # send_zipped_pickle(socket, {'action_sequence': intervention})
        for action in intervention:
            action_dict = {"action": action}
            send_zipped_pickle(socket, action_dict)
            print(
                "Step {}: Sent {} to simulator".format(agent.attempt_count, action_dict)
            )
            # wait for results and append the results to the agent's log
            response = recv_zipped_pickle(socket)
            # todo: this only handles a single trial.
            trial = response["trial"]
            is_current_trial = response["is_current_trial"]
            trial_finished = response["trial_finished"]
            agent.update_trial(trial, is_current_trial)
            print(
                "Step {}: Received action result from simulator".format(
                    agent.attempt_count
                )
            )
            attempt_results = agent.get_last_results()
            agent.pretty_print_last_results()

            if attempt_results is None:
                print("attempt is none")

            # update beliefs, uses agent's log to get executed interventions/outcomes
            max_a_posteriori_chains, num_chains_pruned_action = agent.update_causal_model(
                action, attempt_results
            )
            num_chains_pruned_step += num_chains_pruned_action

            if 0 < len(max_a_posteriori_chains) < 20:
                chains_with_positive_belief = (
                    agent.causal_chain_structure_space.chains_with_positive_belief()
                )
                if len(chains_with_positive_belief) < 20:
                    print(
                        "Step {}: CHAINS WITH POSITIVE BELIEF".format(
                            agent.attempt_count
                        )
                    )
                    agent.causal_chain_structure_space.pretty_print_causal_chain_idxs(
                        chains_with_positive_belief
                    )
                print("Step {}: MAP CHAINS".format(agent.attempt_count))
                agent.causal_chain_structure_space.pretty_print_causal_chain_idxs(
                    max_a_posteriori_chains
                )

            # end attempt if the simulator did
            if response["env_reset"] or trial_finished:
                agent.env_reset()
                break

        agent.verify_true_chains_above_belief_threshold()

        num_chains_pruned.append(num_chains_pruned_step)

        agent.plot_num_pruned(num_chains_pruned)

        if trial_finished:
            agent.finish_trial(test_trial=False)
            trial_finished = False
            trial_idx += 1
            if trial_idx >= len(THREE_LEVER_TRIALS):
                break
            selected_trial = THREE_LEVER_TRIALS[trial_idx]
            send_zipped_pickle(socket, {"set_trial": selected_trial})
            response_str = recv_zipped_pickle(socket)
            print(response_str)

        agent.attempt_count += 1

        # quit when we found all true chains
        if len(chains_with_positive_belief) == len(agent.true_chains) and all(
            chain in agent.true_chains for chain in chains_with_positive_belief
        ):
            break

    return


if __name__ == "__main__":
    main()
