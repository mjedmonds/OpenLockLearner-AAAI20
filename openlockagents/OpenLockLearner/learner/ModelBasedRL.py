import numpy as np
import math
import sys
import random
from collections import defaultdict
import operator
import dill

from openlockagents.OpenLockLearner.learner.ModelBasedPlanner import ModelBasedPlanner

from openlockagents.OpenLockLearner.util.common import (
    get_highest_N_idxs,
    get_lowest_N_idxs,
    get_highest_N_values,
    get_lowest_N_values,
    AblationParams,
    renormalize,
    verify_valid_probability_distribution,
)
from openlockagents.common.common import DEBUGGING

MAX_ENERGY = sys.float_info.max


class ModelBasedRLAgent:
    def __init__(
        self, n_solutions, goal, noise_mu=0.0, noise_std_dev=0.0001, lambda_multiplier=1
    ):
        # state is the number of solutions remaining followed by all solutions executed
        self.state = [n_solutions]
        self.model_based_planner = ModelBasedPlanner(goal)
        self.goal = goal
        self.noise_mu = noise_mu
        self.noise_std_dev = noise_std_dev
        self.lambda_multiplier = lambda_multiplier
        # self.noise_std_dev = 0.1

    def compute_chain_transition_probability(
        self,
        causal_chain_space,
        causal_chain_idx,
        intervention_idxs_executed_set,
        interventions_executed_set,
        first_agent_trial,
        **kwargs
    ):
        ablation = kwargs["ablation"] if "ablation" in kwargs.keys() else None

        bottom_up_chain_belief = causal_chain_space.bottom_up_belief_space.beliefs[
            causal_chain_idx
        ]
        top_down_chain_belief = causal_chain_space.top_down_belief_space.beliefs[
            causal_chain_idx
        ]

        # noise = np.random.normal(self.noise_mu, self.noise_std_dev)
        noise = 0

        causal_chain_actions = causal_chain_space.structure_space.get_actions(
            causal_chain_idx
        )

        # check if this chain is already in our solutions
        if causal_chain_actions in self.state[1:]:
            energy = MAX_ENERGY
        # already executed this causal chain idx
        # elif causal_chain_idx in intervention_idxs_executed:
        #     energy = MAX_ENERGY
        # need to check that we haven't executed this action sequence before
        elif causal_chain_actions in interventions_executed_set:
            energy = MAX_ENERGY
        # check if this chain satisfies our model based planner goal
        elif not self.model_based_planner.determine_chain_satisfies_goal(
            causal_chain_space.structure_space, causal_chain_idx
        ):
            energy = MAX_ENERGY
        else:
            # p(chain) = 1/Z * exp(-energy)
            #          = 1/Z * exp(-1(log(bottom_up_term) + \lambda * log(top_down_term))
            top_down_term = (
                -1 * math.log(top_down_chain_belief)
                if top_down_chain_belief > 0
                else MAX_ENERGY
            )
            bottom_up_term = (
                -1 * math.log(bottom_up_chain_belief)
                if bottom_up_chain_belief > 0
                else MAX_ENERGY
            )
            assert (
                bottom_up_term != 0.0
            ), "Bottom up term has zero belief - chain should be pruned"

            if top_down_term != MAX_ENERGY:
                top_down_term = self.lambda_multiplier * top_down_term

            # ablations
            if ablation.TOP_DOWN:
                top_down_term = 0
            if ablation.TOP_DOWN_FIRST_TRIAL and first_agent_trial:
                top_down_term = 0
            if ablation.BOTTOM_UP:
                bottom_up_term = 0

            energy = bottom_up_term + top_down_term

        final_probability = math.exp(-1 * energy) + noise
        return final_probability
        # return bottom_up_chain_belief

    def marginalize_node_transition_probability(
        self,
        action_beliefs,
        causal_chain_space,
        causal_chain_idx,
        causal_change_idx,
        chain_posterior,
    ):
        causal_chain_actions = causal_chain_space.structure_space.get_actions(
            causal_chain_idx
        )
        causal_chain_attributes = causal_chain_space.structure_space.get_attributes(
            causal_chain_idx
        )
        node_action = causal_chain_actions[causal_change_idx]

        # marginalize the belief of this action by summing over chain posteriors
        action_beliefs[node_action] += chain_posterior

        return action_beliefs

    def random_chain_policy(self, idxs):
        return random.choice(idxs)

    def random_action_policy(self, action_space):
        return random.choice(action_space)

    def greedy_action_policy(
        self,
        causal_chain_space,
        causal_chain_idxs,
        causal_change_idx,
        action_sequence,
        first_agent_trial,
        intervention_idxs_executed=None,
        interventions_executed=None,
        ablation=None,
    ):
        """
        Performs a greedy policy action by action, conditioned on the action sequence executed so far
        :param causal_chain_space: CausalChainSpace
        :param causal_chain_idxs: causal_chain_idxs to consider
        :param causal_change_idx: current causal change index
        :param action_sequence: action sequence executed this attempt
        :param first_agent_trial: is this the agent's first trial?
        :param intervention_idxs_executed: indices of interventions executed this trial
        :param interventions_executed: action sequences executed this trial
        :param ablation: model ablations
        :return: best_action: the optimal action, action_beliefs: the beliefs of each action
        """
        interventions_executed_set = set(interventions_executed)
        intervention_idxs_executed = np.array(intervention_idxs_executed)
        intervention_idxs_executed_set = set(intervention_idxs_executed.flatten())

        # causal_relation_domain_at_index = causal_chain_space.structure_space.subchain_indexed_domains[
        #     causal_change_idx
        # ]

        # find causal chains that contain this action sequence
        if len(action_sequence) > 0:
            causal_chain_idxs_with_action_seq = causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                action_sequence
            )
            causal_chain_idxs = list(
                set(causal_chain_idxs).intersection(causal_chain_idxs_with_action_seq)
            )
            if DEBUGGING:
                self.sub_action_sequence_sanity_checks(
                    causal_chain_space,
                    action_sequence,
                    causal_chain_idxs_with_action_seq,
                    causal_chain_idxs,
                )

        assert len(causal_chain_idxs) > 0, "No causal chains to select action from"
        if DEBUGGING:
            causal_chains_satisfying_goal = [x for x in causal_chain_idxs if self.model_based_planner.determine_chain_satisfies_goal(causal_chain_space.structure_space, x)]
            assert len(causal_chains_satisfying_goal) > 0, "No causal chains satisfy the goal"

        # construct dict of actions to beliefs
        action_beliefs = defaultdict(int)

        # maginalize over chains iterate over all causal chains, and add their respective beliefs to the corresponding action
        for causal_chain_idx in causal_chain_idxs:
            chain_posterior = self.compute_chain_transition_probability(
                causal_chain_space=causal_chain_space,
                causal_chain_idx=causal_chain_idx,
                intervention_idxs_executed_set=intervention_idxs_executed_set,
                interventions_executed_set=interventions_executed_set,
                first_agent_trial=first_agent_trial,
                **{"ablation": ablation}
            )

            # skip marginalizing any action with no posterior
            if chain_posterior == 0.0:
                continue

            action_beliefs = self.marginalize_node_transition_probability(
                action_beliefs,
                causal_chain_space,
                causal_chain_idx,
                causal_change_idx,
                chain_posterior,
            )

        # if we were unable to pick and model-based action, choose a random action
        # this is possible when the agent has sufficient explored, but started an action sequence that corresponds to a solution already found (e.g. common effect)
        # in this case, the only causal chains that satisfy the goal are the solution already executed, but the solution chains also cannot be picked, so we choose a random action (the attempt is wasted)
        if len(action_beliefs.values()) == 0:
            random_chain = self.random_chain_policy(causal_chain_idxs)
            random_chain_actions = causal_chain_space.structure_space.get_actions(random_chain)
            action = random_chain_actions[causal_change_idx]
            return action, {}
            # variables = {
            #     "model_based_agent": self,
            #     "causal_chain_space": causal_chain_space,
            #     "causal_chain_idxs": causal_chain_idxs,
            #     "causal_change_idx": causal_change_idx,
            #     "action_sequence": action_sequence,
            #     "first_agent_trial": first_agent_trial,
            #     "intervention_idxs_executed": intervention_idxs_executed,
            #     "interventions_executed": interventions_executed,
            #     "ablation": ablation,
            # }
            # with open("/home/mark/Desktop/greedy_action_policy_problem.dill", 'wb') as f:
            #     dill.dump(variables, f)

        assert min(action_beliefs.values()) >= 0, "Action beliefs has negative value"

        # renormalize
        normalization_factor = sum(action_beliefs.values())
        for key in action_beliefs.keys():
            action_beliefs[key] /= normalization_factor

        assert verify_valid_probability_distribution(
            action_beliefs.values()
        ), "action beliefs is not a valid probability distribution"
        # best action has max marginal belief
        # assert False, "sum does not equal 1, problem"
        best_action = max(action_beliefs.items(), key=operator.itemgetter(1))[0]
        return best_action, action_beliefs

    def greedy_chain_policy(
        self,
        causal_chain_space,
        causal_chain_idxs,
        first_agent_trial,
        intervention_idxs_executed=None,
        interventions_executed=None,
        ablation=None,
    ):
        """
        Picks the causal_chain_idx with maximal transition probability
        :param causal_chain_space: causal chain space wrapper
        :param causal_chain_idxs: list of indices to choose from
        :return:
        """

        if ablation.TOP_DOWN:
            beliefs_in_use_during_ablation = causal_chain_space.bottom_up_belief_space
        elif ablation.BOTTOM_UP:
            beliefs_in_use_during_ablation = causal_chain_space.top_down_belief_space
        elif ablation.TOP_DOWN_FIRST_TRIAL:
            beliefs_in_use_during_ablation = None
        else:
            beliefs_in_use_during_ablation = None

        # verify belief sources are properly normalized
        assert verify_valid_probability_distribution(
            causal_chain_space.bottom_up_belief_space
        ), "Bottom up belief space not normalized"
        assert verify_valid_probability_distribution(
            causal_chain_space.top_down_belief_space
        ), "Top down belief space not normalized"

        # top_down_idxs_with_positive_belief = [i for i in range(len(causal_chain_space.top_down_belief_space.beliefs)) if causal_chain_space.top_down_belief_space.beliefs[i] > 0]

        interventions_executed_set = set(interventions_executed)
        intervention_idxs_executed_set = set(intervention_idxs_executed)
        chain_transition_probabilities = []
        for causal_chain_idx in causal_chain_idxs:
            # if causal_chain_idx in top_down_idxs_with_positive_belief:
            #     print("top-down idx")
            chain_transition_probabilities.append(
                self.compute_chain_transition_probability(
                    causal_chain_space=causal_chain_space,
                    causal_chain_idx=causal_chain_idx,
                    intervention_idxs_executed_set=intervention_idxs_executed_set,
                    interventions_executed_set=interventions_executed_set,
                    first_agent_trial=first_agent_trial,
                    **{"ablation": ablation}
                )
            )
        chain_transition_probabilities = np.array(chain_transition_probabilities)
        chain_transition_probabilities = renormalize(chain_transition_probabilities)
        assert verify_valid_probability_distribution(
            chain_transition_probabilities
        ), "chain transitions not a valid probability distribution"

        # minimize the energy of the transition
        best_energy_idx = np.argmax(chain_transition_probabilities)
        best_chain_idx = causal_chain_idxs[best_energy_idx]

        assert (
            best_chain_idx not in intervention_idxs_executed
        ), "Picked the same intervention index twice!"

        if beliefs_in_use_during_ablation is not None:
            self.debug_chain_transition_probability(
                causal_chain_idxs,
                causal_chain_space,
                intervention_idxs_executed,
                chain_transition_probabilities,
                beliefs_in_use_during_ablation=beliefs_in_use_during_ablation,
            )

        return best_chain_idx

    def update_state(self, new_state, causal_action_sequence):
        # number of solutions remaining decreased
        if new_state != self.state[0]:
            assert (
                new_state < self.state[0]
            ), "Number of solutions should monotonically decrease"
            self.state[0] = new_state
            self.state.append(causal_action_sequence)

    def debug_chain_transition_probability(
        self,
        causal_chain_idxs,
        causal_chain_space,
        intervention_idxs_executed,
        chain_transition_probabilities,
        beliefs_in_use_during_ablation,
    ):
        # get the best chain indices from top-down, bottom-up, and final belief
        # top down beliefs
        highest_N_top_down_belief_idxs = get_highest_N_idxs(
            100,
            causal_chain_space.top_down_belief_space.beliefs[causal_chain_idxs],
            min_value=0,
        )
        highest_N_top_down_belief_idxs_final_belief = chain_transition_probabilities[
            highest_N_top_down_belief_idxs
        ]
        highest_N_top_down_belief_causal_chain_space_idxs = [
            causal_chain_idxs[i] for i in highest_N_top_down_belief_idxs
        ]
        highest_N_top_down_beliefs = causal_chain_space.top_down_belief_space[
            highest_N_top_down_belief_causal_chain_space_idxs
        ]

        # bottom up beliefs
        highest_N_bottom_up_belief_idxs = get_highest_N_idxs(
            100,
            causal_chain_space.bottom_up_belief_space.beliefs[causal_chain_idxs],
            min_value=0,
        )
        highest_N_bottom_up_belief_idxs_final_belief = chain_transition_probabilities[
            highest_N_bottom_up_belief_idxs
        ]
        highest_N_bottom_up_belief_causal_chain_space_idxs = [
            causal_chain_idxs[i] for i in highest_N_bottom_up_belief_idxs
        ]
        highest_N_bottom_up_beliefs = causal_chain_space.bottom_up_belief_space[
            highest_N_bottom_up_belief_causal_chain_space_idxs
        ]
        highest_N_bottom_up_beliefs_value_to_index_map = self.construct_value_to_idx_dict(
            5, causal_chain_space.bottom_up_belief_space, get_highest_N_values
        )

        # final belief
        highest_N_final_beliefs_idxs = get_highest_N_idxs(
            100, chain_transition_probabilities, min_value=0
        )
        highest_N_final_beliefs = chain_transition_probabilities[
            highest_N_final_beliefs_idxs
        ]
        highest_N_final_beliefs_causal_chain_space_idxs = [
            causal_chain_idxs[i] for i in highest_N_final_beliefs_idxs
        ]
        full_chain_transition_probabilities = np.full(
            len(causal_chain_space.bottom_up_belief_space.beliefs), fill_value=0.0
        )
        full_chain_transition_probabilities[
            causal_chain_idxs
        ] = chain_transition_probabilities
        highest_N_final_belief_idx_bottom_up_beliefs = causal_chain_space.bottom_up_belief_space[
            highest_N_final_beliefs_causal_chain_space_idxs
        ]
        highest_N_final_belief_idx_top_down_beliefs = causal_chain_space.top_down_belief_space[
            highest_N_final_beliefs_causal_chain_space_idxs
        ]
        highest_N_final_belief_value_to_index_map = self.construct_value_to_idx_dict(
            5, full_chain_transition_probabilities, get_highest_N_values
        )

        if len(highest_N_final_beliefs) == 0:
            print("pause")

        # initial sanity checks
        assert len(highest_N_final_beliefs) != 0, "No final beliefs have belief above 0"
        assert all(
            [
                highest_N_bottom_up_belief > 0
                for highest_N_bottom_up_belief in highest_N_top_down_beliefs
            ]
        ), "highest bottom-up beliefs contain 0 belief - rest of checks may fail"
        assert all(
            [
                highest_N_final_belief > 0
                for highest_N_final_belief in highest_N_final_beliefs
            ]
        ), "highest final beliefs contain 0 belief - rest of checks may fail"
        assert all(
            [
                highest_N_top_down_belief > 0
                for highest_N_top_down_belief in highest_N_top_down_beliefs
            ]
        ), "highest top-down beliefs contain 0 belief - rest of checks may fail"

        # verify that the ablated belief source matches the chain transition probabilities everywhere the chain transition probabilities are non-zero. If they are zero, the chain either has been already executed or does not satisfy the goal
        chain_transition_probabilities_idxs_with_positive_belief = np.where(
            chain_transition_probabilities > 0
        )[0]
        causal_graph_space_idxs_with_positive_belief = np.array(causal_chain_idxs)[
            chain_transition_probabilities_idxs_with_positive_belief
        ]
        assert len(causal_graph_space_idxs_with_positive_belief) == len(
            chain_transition_probabilities_idxs_with_positive_belief
        ), "Lengths should be the same"
        # check for small tolerance difference between the original source beliefs and the computed beliefs
        assert all(
            [
                abs(
                    beliefs_in_use_during_ablation[
                        causal_graph_space_idxs_with_positive_belief[i]
                    ]
                    - chain_transition_probabilities[
                        chain_transition_probabilities_idxs_with_positive_belief[i]
                    ]
                )
                < 1e-10
                for i in range(len(causal_graph_space_idxs_with_positive_belief))
            ]
        ), "Computed chain transition indices do not match source belief under ablation where goal is satisfied and indices have not been executed"

        # verify that all causal chains with highest belief satisfy goal
        assert all(
            [
                self.model_based_planner.determine_chain_satisfies_goal(
                    causal_chain_space.structure_space, x
                )
                for x in highest_N_final_beliefs_causal_chain_space_idxs
            ]
        ), "not all chains with highest final belief satisfy model-based goal"

        # verify correspondence between model, highest beliefs, and energies
        # this should only be used in an ablation mode, as the interplay between two terms can mean the energy does not match the belief of either top-down/bottom-up
        self.check_correspondence_between_belief_energy_and_model_under_ablation(
            causal_chain_space.structure_space,
            beliefs_in_use_during_ablation,
            highest_N_final_beliefs_causal_chain_space_idxs,
            intervention_idxs_executed,
        )

        zipped_beliefs = list(
            zip(
                causal_chain_space.bottom_up_belief_space.beliefs,
                causal_chain_space.top_down_belief_space.beliefs,
            )
        )

        # print the top N
        num_to_print = 10
        print("Top top-down chains")
        causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            highest_N_top_down_belief_idxs[:num_to_print],
            beliefs=zipped_beliefs,
            energies=full_chain_transition_probabilities,
            belief_label="(bottom-up,top-down)",
        )
        print("Top bottom-up chains")
        causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            highest_N_bottom_up_belief_idxs[:num_to_print],
            beliefs=zipped_beliefs,
            energies=full_chain_transition_probabilities,
            belief_label="(bottom-up,top-down)",
        )
        print("Top final belief chains")
        causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            highest_N_final_beliefs_idxs[:num_to_print],
            beliefs=zipped_beliefs,
            energies=full_chain_transition_probabilities,
            belief_label="(bottom-up,top-down)",
        )

    def construct_value_to_idx_dict(self, N, source_arr, getter_func):
        # find best criteria values through a set of the source array
        best_unique_values = getter_func(N, np.array(list(set(source_arr))))
        best_value_to_idx_map = defaultdict(set)
        for best_unique_value in best_unique_values:
            source_arr_idxs = np.argwhere(source_arr == best_unique_value)
            best_value_to_idx_map[best_unique_value].update(
                source_arr_idxs.flatten().tolist()
            )
        return best_value_to_idx_map

    def check_correspondence_between_belief_energy_and_model_under_ablation(
        self,
        causal_chain_space,
        beliefs,
        highest_final_belief_idxs,
        intervention_idxs_executed,
    ):
        """
        checks that the highest beliefs that satisfy the model-based goals also have the lowest energy. This should be run under an ablation, or run with a probability distribution that combines top-down and bottom-up beliefs
        :param causal_chain_space: space of causal chains
        :param beliefs: top-down or bottom-up beliefs
        :param lowest_energy_idxs: indices of the lowest energy causal chains
        :return:
        """
        belief_to_index_map = self.construct_value_to_idx_dict(
            5, beliefs, get_highest_N_values
        )
        # todo parameterize to sort by highest or lowest
        # sort the belief_to_index_map keys from highest to lowest
        sorted_keys = sorted(belief_to_index_map.keys(), reverse=True)
        for key in sorted_keys:
            idxs_with_this_belief = belief_to_index_map[key]
            # no intersection between this belief and the lowest energy. All of these indices must not satisfy goal
            idxs_that_satisfy_goal = self.get_idxs_that_satisfy_goal(
                causal_chain_space.structure, idxs_with_this_belief
            )
            # remove any idxs that have already been executed
            idxs_that_satisfy_goal = [
                x for x in idxs_that_satisfy_goal if x not in intervention_idxs_executed
            ]

            intersection_with_highest_final_belief_idxs = idxs_with_this_belief.intersection(
                set(highest_final_belief_idxs)
            )
            # verify that the number of idxs that satisfy the goal match the number of idxs with the highest final belief
            # reasoning/explanation: since we are under an ablation, only a single belief is contributing to the final belief. This means the highest belief that satisfies the goal should also have the highest final belief. We we check for this
            assert_str = "Belief not part of highest final belief, but satisfies goal and has highest belief under ablation"
            try:
                assert intersection_with_highest_final_belief_idxs.issubset(
                    set(idxs_that_satisfy_goal)
                ), assert_str
            except AssertionError:
                raise AssertionError(assert_str)
            # we found and verified that lowest energy idxs correspond to the highest belief that satisfies the goal
            if intersection_with_highest_final_belief_idxs:
                break

    def all_idxs_satisfy_goal(self, causal_chain_structure_space, causal_chain_idxs):
        return all(
            [
                self.model_based_planner.determine_chain_satisfies_goal(
                    causal_chain_structure_space, causal_chain_idx
                )
                for causal_chain_idx in causal_chain_idxs
            ]
        )

    def any_idxs_satisfy_goal(self, causal_chain_structure_space, causal_chain_idxs):
        return any(
            [
                self.model_based_planner.determine_chain_satisfies_goal(
                    causal_chain_structure_space, causal_chain_idx
                )
                for causal_chain_idx in causal_chain_idxs
            ]
        )

    def get_idxs_that_satisfy_goal(
        self, causal_chain_structure_space, causal_chain_idxs
    ):
        return [
            x
            for x in causal_chain_idxs
            if self.model_based_planner.determine_chain_satisfies_goal(
                causal_chain_structure_space, x
            )
        ]

    def sub_action_sequence_sanity_checks(
        self,
        causal_chain_space,
        causal_action_sequence,
        causal_chain_idxs_with_action_seq,
        causal_chain_idxs,
    ):
        action_sub_sequences = [
            causal_chain_space.structure_space.get_actions(x)[
                : len(causal_action_sequence)
            ]
            for x in causal_chain_idxs_with_action_seq
        ]
        causal_action_sequence_tuple = tuple(causal_action_sequence)
        assert all(
            [
                action_sub_sequence == causal_action_sequence_tuple
                for action_sub_sequence in action_sub_sequences
            ]
        ), "Fetched action sequence that does not match already-executed action sequence"
        assert all(
            [
                causal_chain_space.top_down_belief_space[causal_chain_idxs[x]] > 0
                for x in range(len(causal_chain_idxs))
            ]
        ), "top down beliefs considered are not positive"
        assert all(
            [
                causal_chain_space.bottom_up_belief_space[causal_chain_idxs[x]] > 0
                for x in range(len(causal_chain_idxs))
            ]
        ), "bottom up beliefs considered are not positive"
