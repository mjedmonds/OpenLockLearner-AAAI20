from collections import defaultdict
import copy
import time

import numpy as np

from openlockagents.OpenLockLearner.util.common import (
    print_message,
)

class QLearner:
    def __init__(self, max_num_solutions, causal_chains, discount_factor, alpha):
        # q function is accessed by state, indices correspond to actions, values at indices are Q(s,a)
        self.num_solutions = max_num_solutions
        self.global_Q = np.zeros((max_num_solutions + 1, len(causal_chains)))
        # outer index is trial; inner index is num_solutions_remaining (state). indices correspond to actions, values at indices are Q(s,a)
        self.local_Q = defaultdict(
            lambda: np.zeros((max_num_solutions + 1, len(causal_chains)))
        )

        self.discount_factor = discount_factor
        self.alpha = alpha
        self.rewards = []

    def initialize_local_Q(self, trial_name):
        # todo: can we do something better than deleting the old local Q? Perhaps blend old one with the latest from the global?
        if trial_name in self.local_Q.keys():
            del self.local_Q[trial_name]

    def td_update(self, Q, state, action, reward, next_state):
        best_next_action = self.get_best_action(Q, next_state)
        td_target = reward + self.discount_factor * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        td_contribution = self.alpha * td_delta
        Q[state][action] += td_contribution
        return td_contribution

    def global_td_update(self, state, action, reward, next_state):
        td_contribution = self.td_update(
            self.global_Q, state, action, reward, next_state
        )
        self.rewards.append(reward)
        return td_contribution

    def local_td_update(self, trial_name, state, action, reward, next_state):
        self.td_update(self.local_Q[trial_name], state, action, reward, next_state)

    def initialize_local_q(self, trial_name):
        start_time = time.time()
        print_message(
            self.trial_count,
            self.attempt_count,
            "Initializing new Q for {}".format(trial_name),
        )
        self.qlearner.initialize_local_Q(trial_name)
        if self.qlearner.global_Q.max() == 0.0:
            # todo: this is stub code to get default dict to be created
            dummy = self.qlearner.local_Q[trial_name].shape
            return

        indexed_confidences, total_confidences = self.compute_indexed_confidences(
            trial_name
        )

        total_possible_confidence = 0
        for causal_chain in self.causal_chain_space.causal_chains:
            chain_confidence = 0
            total_possible_chain_confidence = 0
            for causal_chain_prime in self.causal_chain_space.causal_chains:
                # todo: refactor to use attributes only rather than state + attributes
                for i in range(len(causal_chain_prime.attributes)):
                    # color confidence
                    color_confidence = indexed_confidences[i]["color"]
                    total_possible_chain_confidence += color_confidence
                    # contributes if chain attributes match (chain distance measure)
                    if causal_chain.attributes[i] == causal_chain_prime.attributes[i]:
                        chain_confidence += color_confidence
                        # chain_confidence += color_confidence / total_color_confidence
                    # position confidence
                    position_confidence = indexed_confidences[i]["position"]
                    total_possible_chain_confidence += position_confidence
                    # contributes if chain states (positions) match (chain distance measure)
                    if causal_chain.states[i] == causal_chain_prime.states[i]:
                        chain_confidence += position_confidence
                        # chain_confidence += position_confidence / total_position_confidence

                # assign the raw chain confidence; confidence is constant across solutions remaining - still need to multiply by global Q value
                for num_solutions_remaining in range(self.qlearner.global_Q.shape[0]):
                    # normalize and multiply by the global Q value
                    # self.qlearner.local_Q[trial_name][num_solutions_remaining][causal_chain.chain_id] = (chain_confidence / total_possible_chain_confidence) * self.qlearner.global_Q[num_solutions_remaining][causal_chain.chain_id]
                    # add the proportion of the g' Q-value to g according to confidence
                    self.qlearner.local_Q[trial_name][num_solutions_remaining][
                        causal_chain.chain_id
                    ] += (
                        chain_confidence
                        * self.qlearner.global_Q[num_solutions_remaining][
                            causal_chain_prime.chain_id
                        ]
                    )
                    # self.qlearner.local_Q[trial_name][num_solutions_remaining][causal_chain.chain_id] = chain_confidence

            total_possible_confidence += total_possible_chain_confidence

        # normalize by the total possible confidence
        # for num_solutions_remaining in self.qlearner.local_Q[trial_name].keys():
        #     self.qlearner.local_Q[trial_name][num_solutions_remaining] = self.qlearner.local_Q[trial_name][num_solutions_remaining] / total_possible_confidence
        print_message(
            self.trial_count,
            self.attempt_count,
            "Initializing new Q for {} took {:0.6f}s".format(
                trial_name, time.time() - start_time
            ),
        )

    def initialize_local_q2(self, trial_name):
        start_time = time.time()
        print_message(
            self.trial_count,
            self.attempt_count,
            "Initializing new Q for {}".format(trial_name),
        )
        self.qlearner.initialize_local_Q(trial_name)
        if self.qlearner.global_Q.max() == 0.0:
            # todo: refactor; this is stub code to get default dict to be created
            dummy = self.qlearner.local_Q[trial_name].shape
            return

        # copy global Q to initialize local Q
        self.qlearner.local_Q[trial_name] = copy.deepcopy(self.qlearner.global_Q)

        print_message(
            self.trial_count,
            self.attempt_count,
            "Initializing new Q for {} took {:0.6f}s".format(
                trial_name, time.time() - start_time
            ),
        )

    def compute_chain_similarity(self, chain1, chain2, indexed_confidences=None):
        if indexed_confidences is None:
            trial_name = self.env.cur_trial.name
            indexed_confidences, _ = self.compute_indexed_confidences(trial_name)
        # todo: consolidate position and color into attributes
        similarity = 0
        total_similarity_possible = 0
        for i in range(len(chain1.states)):
            # indicator function
            if chain1.states[i] == chain2.states[i]:
                # similarity contribution is based on confidence of attribute
                similarity += indexed_confidences[i]["position"]
            total_similarity_possible += indexed_confidences[i]["position"]
        for i in range(len(chain1.attributes)):
            # indicator function
            if chain1.attributes[i] == chain2.attributes[i]:
                # similarity contribution is based on confidence of attribute
                similarity += indexed_confidences[i]["color"]
            total_similarity_possible += indexed_confidences[i]["color"]
        # similarity += sum(indexed_confidences[i]['color'] for i in range(len(chain1.attributes)) if chain1.attributes[i] == chain2.attributes[i])
        return similarity / total_similarity_possible

    def compute_indexed_confidences(self, trial_name=None):
        if trial_name is None:
            trial_name = self.env.cur_trial.name
        # precompute confidences; they are specific to each attribute at each index
        num_indices = len(self.causal_chain_space.causal_chains[0].states)
        indexed_confidences = []
        total_confidences = []
        total_color_confidence = 0
        total_position_confidence = 0
        # todo: make this more robust/general to handle more than two attributes
        for i in range(num_indices):
            distribution = self.attribute_space.local_attributes[
                trial_name
            ].get_distribution_at_index(i)
            color_confidence = distribution["color"].compute_confidence()
            total_color_confidence += color_confidence
            position_confidence = distribution["position"].compute_confidence()
            total_position_confidence += position_confidence
            indexed_confidences.append(
                {"color": color_confidence, "position": position_confidence}
            )

        total_confidences.append(
            {"color": total_color_confidence, "position": total_position_confidence}
        )

        return indexed_confidences, total_confidences

    def remove_causal_chain_from_local_Q(self, trial_name, chain_idx):
        """
        Removes causal chain from Q by marking its index as 0
        :param chain_idx:
        :return:
        """
        for state in range(self.local_Q[trial_name].shape[0]):
            self.local_Q[trial_name][state][chain_idx] = 0

    @staticmethod
    def get_best_action(Q, state):
        best_action = np.argmax(Q[state])
        assert (
            Q[state][best_action] >= 0
        ), "Best action has negative Q-value, indicating chain was pruned"
        return best_action

    @staticmethod
    def get_greedy_policy(Q):
        greedy_policy = np.zeros(Q.shape[0], dtype=int)
        greedy_q_values = np.zeros(Q.shape[0])
        for state in range(Q.shape[0]):
            greedy_policy[state] = QLearner.get_best_action(Q, state)
            greedy_q_values[state] = Q[state][greedy_policy[state]]
        return greedy_policy, greedy_q_values

    def get_greedy_global_policy(self):
        return self.get_greedy_policy(self.global_Q)

    def get_greedy_local_policy(self, trial_name):
        return self.get_greedy_policy(self.local_Q[trial_name])

    def get_top_k_actions_policy(self, Q, k):
        top_k_actions = []
        top_k_q_values = []
        for state in range(Q.shape[0]):
            # gets indices of top k actions in Q[state]
            top_actions = np.argpartition(Q[state], -k)[-k:]
            top_actions = np.flipud(top_actions[np.argsort(Q[state][top_actions])])
            top_k_actions.append(top_actions)
            top_q_values = Q[state][top_actions[:]]
            top_k_q_values.append(top_q_values)
        return top_k_actions, top_k_q_values

    def get_top_k_actions_global_policy(self, k):
        return self.get_top_k_actions_policy(self.global_Q, k)

    def get_top_k_actions_local_policy(self, trial_name, k):
        return self.get_top_k_actions_policy(self.local_Q[trial_name], k)
