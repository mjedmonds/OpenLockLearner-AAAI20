import random
import itertools
import numpy as np
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import CausalRelationType


class State:
    def __init__(self, state_id, state_value):
        self.id = state_id  # id/label for the state
        self.value = state_value  # value fo the state,

    def __str__(self):
        return self.id + ": " + str(self.value)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.id == other
        if isinstance(other, State):
            return self.id == other.id and self.value == other.value


class Outcome:
    def __init__(self, state_ids, state_assignments):
        self.state_ids = []
        self.state_values = []
        for s_id, value in zip(state_ids, state_assignments):
            self.state_ids.append(s_id)
            self.state_values.append(value)
        self.state_ids = tuple(self.state_ids)
        self.state_values = tuple(self.state_values)

    def __str__(self):
        return str(list(zip(self.state_ids, self.state_values)))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            self.state_ids == other.state_ids
            and self.state_values == other.state_values
        )

    # self is current, other is consider previous
    def __sub__(self, other):
        differences = []
        for idx in range(len(self.state_ids)):
            state_id = self.state_ids[idx]
            state_value = self.state_values[idx]
            other_idx = other.state_ids.index(state_id)
            other_state_value = other.state_values[other_idx]
            # compute fluent change
            diff = state_value - other_state_value
            # no change
            if diff == 0:
                continue
            # convert to causal relation type
            causal_relation_type = (
                CausalRelationType.zero_to_one
                if diff == 1
                else CausalRelationType.one_to_zero
            )
            differences.append((state_id, causal_relation_type))
        return differences

    @staticmethod
    def parse_results_into_outcome(results, idx=-1):
        agent_idx = results[0].index("agent")
        state_ids = results[0][1 : agent_idx + 1]
        state_values = results[idx][1 : agent_idx + 1]
        return Outcome(state_ids, state_values)


class OutcomeSpace:
    def __init__(self, state_space, num_states_in_chain, using_ids=True):
        self.outcomes = self.generate_outcome_space(state_space, num_states_in_chain)
        # todo: refactor. Boolean to represent whether or not we are using an ID or string representation of outcomes
        self.using_ids = using_ids

    # observation space is all possible combinations of states
    # can observe on one variable, two variable, or three
    @staticmethod
    def generate_outcome_space(state_space, num_states_in_chain):
        outcome_space = []
        # generate all possible ways of choosing binary values for every state variable in the chain
        # state_permutations = list(itertools.permutations(state_space, r=num_states_in_chain))
        state_combinations = list(
            itertools.product(state_space, repeat=num_states_in_chain)
        )
        binary_combinations = list(
            itertools.product([0, 1], repeat=num_states_in_chain)
        )
        # combine all state/binary combinations
        for state in state_combinations:
            for binary in binary_combinations:
                outcome_space.append(Outcome(state, binary))

        return np.array(outcome_space)
