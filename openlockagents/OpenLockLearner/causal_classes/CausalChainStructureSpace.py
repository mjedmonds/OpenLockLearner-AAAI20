import numpy as np
import random
import texttable
import jsonpickle
import time
import copy
from operator import itemgetter
from collections import defaultdict

from openlockagents.OpenLockLearner.util.common import (
    check_for_duplicates,
    SANITY_CHECK_ELEMENT_LIMIT,
    ALL_CAUSAL_CHAINS,
)
from openlockagents.common.io.log_io import pretty_write

from openlock.logger_env import ActionLog


class CausalChainStructureSpace:
    def __init__(
        self,
        causal_relation_space,
        chain_length,
        attributes,
        structure,
        attribute_order,
        lever_index_mode,
        print_messages=True
    ):
        # represents every possible permutation of conditional probability tables for each chain
        # self.conditional_probability_table_combinations = None

        self.causal_relation_space = copy.deepcopy(causal_relation_space)
        self.chain_length = chain_length
        self.attributes = attributes
        self.structure = structure
        self.attribute_order = copy.copy(attribute_order)
        self.lever_index_mode = lever_index_mode
        self.state_index_in_attributes = self.attribute_order.index(
            self.lever_index_mode
        )
        self.print_messages = print_messages

        assert len(self.attribute_order) == len(
            attributes
        ), "Attribute order and attributes have different lengths"

        # self.base_schema = self.construct_base_schema(structure)

        self.true_chains = []
        self.true_chain_idxs = []

        self.using_ids = False

        self.subchain_indexed_domains = [
            self.causal_relation_space.causal_relations_no_parent
        ]
        for i in range(1, self.chain_length):
            self.subchain_indexed_domains.append(
                self.causal_relation_space.causal_relations_parent
            )

        self.causal_chains = self.generate_chains()

        self.subchain_indexed_causal_relation_to_chain_index_map = self.construct_chain_indices_by_subchain_index(
            self.causal_chains, self.chain_length
        )

    def construct_chain_indices_by_subchain_index(self, chains, num_subchains):
        chain_indices_by_subchain_index = [
            defaultdict(set) for i in range(num_subchains)
        ]
        for chain_index in range(len(chains)):
            chain = chains[chain_index]
            for subchain_index in range(len(chain)):
                causal_relation = chain[subchain_index]
                chain_indices_by_subchain_index[subchain_index][causal_relation].add(
                    chain_index
                )
        return chain_indices_by_subchain_index

    def generate_chains(self):
        subchain_indexed_domains = [list(x) for x in self.subchain_indexed_domains]
        chains = []
        rejected_chains = []
        counter = 0
        total_num_chains = int(
            np.prod([len(chain_domain) for chain_domain in subchain_indexed_domains])
        )
        # generate all possible chains
        for i in range(len(subchain_indexed_domains[0])):
            root_subchain = subchain_indexed_domains[0][i]
            counter = self.recursive_chain_generation(
                root_subchain,
                [i],
                subchain_indexed_domains,
                depth=1,
                chains=chains,
                rejected_chains=rejected_chains,
                counter=counter,
            )
            print(
                "{}/{} chains generated. {} valid chains".format(
                    counter, total_num_chains, len(chains)
                )
            )
        print(
            "{}/{} chains generated. {} valid chains".format(
                counter, total_num_chains, len(chains)
            )
        )
        return chains

    def recursive_chain_generation(
        self,
        parent_subchain,
        parent_indices,
        chain_domains,
        depth,
        chains,
        rejected_chains,
        counter,
    ):
        terminal_depth = depth == len(chain_domains) - 1
        for i in range(len(chain_domains[depth])):
            child_subchain = chain_domains[depth][i]
            local_parent_indices = copy.copy(parent_indices)
            # verify postcondition of parent matches precondition of child
            if (
                parent_subchain.attributes,
                parent_subchain.causal_relation_type[1],
            ) == child_subchain.precondition:
                # if we aren't at the last chain domain, continue recursing
                if not terminal_depth:
                    local_parent_indices.append(i)
                    counter = self.recursive_chain_generation(
                        child_subchain,
                        local_parent_indices,
                        chain_domains,
                        depth + 1,
                        chains,
                        rejected_chains,
                        counter,
                    )
                # if we are at the terminal depth, this is the final check, add the chain to chains
                else:
                    # collect all parents
                    chain = [
                        chain_domains[j][local_parent_indices[j]]
                        for j in range(len(local_parent_indices))
                    ]
                    chain.append(child_subchain)
                    chains.append(tuple(chain))
                counter += 1
            else:
                if not terminal_depth:
                    counter += len(chain_domains[depth + 1])
                else:
                    # chain = [chain_domains[j][local_parent_indices[j]] for j in range(len(local_parent_indices))]
                    # chain.append(child_subchain)
                    # rejected_chains.append(chain)
                    counter += 1
        return counter

    def __getitem__(self, item):
        # handle slices
        if isinstance(item, slice):
            slice_result = CausalChainStructureSpace(
                self.causal_relation_space,
                self.chain_length,
                self.attributes,
                self.structure,
                self.attribute_order,
                self.lever_index_mode,
            )
            slice_result.causal_chains = self.causal_chains[item]
            return slice_result
        # handle slicing by an arbitrary list of indices
        elif isinstance(item, list) or isinstance(item, np.ndarray):
            slice_result = CausalChainStructureSpace(
                self.causal_relation_space,
                self.chain_length,
                self.attributes,
                self.structure,
                self.attribute_order,
                self.lever_index_mode,
            )
            slice_result.causal_chains = itemgetter(*self.causal_chains)(item)
            return slice_result
        # handle integer access
        elif isinstance(item, int) or np.issubdtype(item, np.integer):
            return self.causal_chains[item]
        else:
            raise TypeError("Invalid argument type")

    def __len__(self):
        return len(self.causal_chains)

    def _size(self):
        return len(self.causal_chains)

    def index(self, item, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.causal_chains)
        for i in range(start, end):
            if self.causal_chains[i] == item:
                return i

    @property
    def num_subchains_in_chain(self):
        if len(self.causal_chains) == 0:
            return 0
        else:
            return len(self.causal_chains[0])

    def append(self, causal_chain):
        self.causal_chains.append(causal_chain)

    def extend(self, causal_chain_manager):
        self.causal_chains.extend(causal_chain_manager.causal_chains)

    def find_causal_chain_idxs(self, target_chain):
        # matching_chains = []
        matching_chain_idxs = []
        for i in range(len(self.causal_chains)):
            if self.causal_chains[i] == target_chain:
                # matching_chains.append(self.causal_chains[i])
                matching_chain_idxs.append(i)

        return matching_chain_idxs

    def find_causal_chain_idxs_with_actions(self, actions):
        # setup the constraints - want to include
        inclusion_constraints = []
        for subchain_idx in range(len(actions)):
            inclusion_constraints.append({"action": actions[subchain_idx]})
        # search for the causal chains satisfying all of the constraints
        matching_chain_idxs = self.find_all_causal_chains_satisfying_constraints(
            inclusion_constraints
        )
        return matching_chain_idxs

    def find_all_causal_chains_satisfying_constraints(
        self, inclusion_constraints, exclusion_constraints=None
    ):
        if exclusion_constraints is None:
            exclusion_constraints = set()

        causal_chain_indices_satisfying_constraints_at_subchain_indices = [
            [] for i in range(len(self.subchain_indexed_domains))
        ]
        # collection relations that satisfy constraints at each index
        for subchain_index in range(len(inclusion_constraints)):
            # if we have a constraint, find causal relations that adhere to the constraints
            if len(inclusion_constraints[subchain_index]) > 0:
                # todo: refactor these to not directly access member classes
                causal_relations_satisfying_constraints = self.causal_relation_space.find_relations_satisfying_constraints(
                    self.subchain_indexed_domains[subchain_index],
                    inclusion_constraints=inclusion_constraints[subchain_index],
                    exclusion_constraints=exclusion_constraints,
                )
                # constraints constitute all causal chains
                if len(causal_relations_satisfying_constraints) == len(
                    self.subchain_indexed_domains[subchain_index]
                ):
                    causal_chain_indices_satisfying_constraints_at_subchain_indices[
                        subchain_index
                    ] = ALL_CAUSAL_CHAINS
                    continue

                # now we have a list of all relations that satisfy the constraint at this subchain index
                # find the causal chain indices that have these relations at this subchain index
                causal_chain_indices_satisfying_constraints_at_subchain_indices[
                    subchain_index
                ] = self.find_causal_chain_indices_satisfying_constraints_at_subchain_index(
                    subchain_index, causal_relations_satisfying_constraints
                )
            else:
                # remove all exclusion constraints from possible relations at this index
                causal_relations_satisfying_constraints = (
                    self.subchain_indexed_domains[subchain_index]
                    - exclusion_constraints
                )
                causal_chain_indices_satisfying_constraints_at_subchain_indices[
                    subchain_index
                ] = self.find_causal_chain_indices_satisfying_constraints_at_subchain_index(
                    subchain_index, causal_relations_satisfying_constraints
                )
        # if inclusion_constraints did not specify constraints for every subchain, find the remaining
        for subchain_index in range(
            len(inclusion_constraints), len(self.subchain_indexed_domains)
        ):
            causal_relations_satisfying_constraints = self.causal_relation_space.find_relations_satisfying_constraints(
                self.subchain_indexed_domains[subchain_index],
                inclusion_constraints=None,
                exclusion_constraints=exclusion_constraints,
            )
            causal_chain_indices_satisfying_constraints_at_subchain_indices[
                subchain_index
            ] = self.find_causal_chain_indices_satisfying_constraints_at_subchain_index(
                subchain_index, causal_relations_satisfying_constraints
            )

        # the final list of causal chain indices is the intersection of all indices satisfy constraints at each subchain index
        final_set_of_causal_chain_indices = set()
        all_subchain_indices_free = True
        optimal_subchain_order = np.argsort(
            [
                len(x)
                for x in causal_chain_indices_satisfying_constraints_at_subchain_indices
            ]
        )
        for subchain_index in optimal_subchain_order:
            # this subchain index has a constraint
            if (
                causal_chain_indices_satisfying_constraints_at_subchain_indices[
                    subchain_index
                ]
                != ALL_CAUSAL_CHAINS
            ):
                all_subchain_indices_free = False
                # if we already have values in the final set, take intersection
                if bool(final_set_of_causal_chain_indices):
                    final_set_of_causal_chain_indices = final_set_of_causal_chain_indices.intersection(
                        causal_chain_indices_satisfying_constraints_at_subchain_indices[
                            subchain_index
                        ]
                    )
                # otherwise start final set with instantiated
                else:
                    final_set_of_causal_chain_indices = causal_chain_indices_satisfying_constraints_at_subchain_indices[
                        subchain_index
                    ]

        if all_subchain_indices_free:
            # if every subchain index was free, return full range of causal chains
            return set(range(len(self.causal_chains)))
        else:
            return set(final_set_of_causal_chain_indices)

    def find_chain_indices_using_causal_relation_at_subchain_index(
        self, subchain_index, causal_relation
    ):
        return self.subchain_indexed_causal_relation_to_chain_index_map[subchain_index][
            causal_relation
        ]

    def find_causal_chain_indices_satisfying_constraints_at_subchain_index(
        self, subchain_index, causal_relations_satisfying_constraints
    ):
        causal_chain_indices_satisfying_constraints_at_subchain_index = set()
        for causal_relation in causal_relations_satisfying_constraints:
            causal_chain_indices_satisfying_constraints_at_subchain_index.update(
                self.find_chain_indices_using_causal_relation_at_subchain_index(
                    subchain_index, causal_relation
                )
            )
        return causal_chain_indices_satisfying_constraints_at_subchain_index

    def set_chains(self, causal_chains):
        self.causal_chains = copy.copy(causal_chains)

    def pop(self, pos=-1):
        return self.causal_chains.pop(pos)

    def clear(self):
        self.causal_chains.clear()

    def reset(self):
        self.true_chains.clear()
        self.true_chain_idxs.clear()

    def equals(self, item1, item2):
        return item1 == item2

    def chain_equals(self, idx, other):
        return self.equals(self.causal_chains[idx], other)

    def get_chain_idx(self, causal_chain):
        return self.causal_chains.index(causal_chain)

    def get_outcome(self, index):
        return tuple([x.causal_relation_type[1] for x in self.causal_chains[index]])

    def get_actions(self, index):
        # todo: 0 is hacked in here, need a way to index by the attribute we are indexing state on
        return tuple([x.action for x in self.causal_chains[index]])

    def get_attributes(self, index):
        return tuple([x.attributes for x in self.causal_chains[index]])

    def remove_chains(self, chain_idxs_to_remove):
        for index in chain_idxs_to_remove:
            self.causal_chains.pop(index)

    def shuffle(self):
        random.shuffle(self.causal_chains)
        self.subchain_indexed_causal_relation_to_chain_index_map = self.construct_chain_indices_by_subchain_index(
            self.causal_chains, self.chain_length
        )

    @property
    def num_attributes(self):
        return len(self.attribute_order)

    def set_true_causal_chains(self, true_chains, belief_space):
        """
        sets the true chains for the chain space based on true_chains
        :param true_chains: list of CompactCausalChains representing the true solutions/causally plausible chains
        :return: nothing
        """
        t = time.time()
        print("Setting true causal chains...")
        self.true_chains = true_chains
        self.true_chain_idxs = []
        for true_chain in self.true_chains:
            chain_idx = self.causal_chains.index(true_chain)
            self.true_chain_idxs.append(chain_idx)

        assert (
            len(self.true_chains) == len(self.true_chain_idxs)
            and None not in self.true_chain_idxs
        ), "Could not find all true chain indices in causal chain space"
        print(
            "Setting true causal chains took {}s. True causal chains: ".format(
                time.time() - t
            )
        )
        self.pretty_print_causal_chain_idxs(self.true_chain_idxs, belief_space, print_messages=self.print_messages)

    def check_for_duplicate_chains(self):
        check_for_duplicates(self.causal_chains)

    def delete_causal_chains(self):
        if hasattr(self, "causal_chains"):
            del self.causal_chains

    def sample_chains(
        self, causal_chain_idxs, sample_size=None, action_sequences_executed=None
    ):

        # if we have no sample_size, sample all possible
        chain_sample_size = (
            len(causal_chain_idxs)
            if sample_size is None
            else min(len(causal_chain_idxs), sample_size)
        )

        assert (
            chain_sample_size != 0
        ), "Chain sample size is 0! No chains would be sampled"

        all_chains_executed = False

        # sample chains
        sampled_causal_chain_idxs = set()

        # randomly pick chains from selected list
        new_idxs = random.sample(range(len(causal_chain_idxs)), chain_sample_size)
        new_idxs = itemgetter(*new_idxs)(causal_chain_idxs)
        # prevent picking the same intervention twice
        if action_sequences_executed is not None and len(action_sequences_executed) > 0:
            new_idxs = [
                new_idxs[i]
                for i in range(len(new_idxs))
                if self.get_actions(new_idxs[i]) not in action_sequences_executed
            ]
            if len(new_idxs) == 0:
                all_chains_executed = True
        if isinstance(new_idxs, int):
            sampled_causal_chain_idxs.add(new_idxs)
        else:
            sampled_causal_chain_idxs.update(new_idxs)

        return (list(sampled_causal_chain_idxs), all_chains_executed)

    def check_if_chain_adheres_to_causal_relations(
        self, causal_chain, causal_relations
    ):
        chain_adheres = True  # assume chain adheres to constraints, prove otherwise

        for causal_relation in causal_relations:

            if not self.check_if_chain_adheres_to_relation(
                causal_chain, causal_relation
            ):
                chain_adheres = False
                break
        return chain_adheres

    def check_if_chain_adheres_to_relation(self, causal_chain, causal_relation):
        """
        This function checks if the chain contains a matching state, action and attribute.
        If all of the above match, and the causal_relation_type does not match, this chain
        does not adhere to the relation
        :param causal_chain:
        :param causal_relation:
        :return:
        """
        causal_relation_action = causal_relation.action
        causal_relation_type = causal_relation.causal_relation_type
        causal_relation_attributes = causal_relation.attributes

        chain_actions = causal_chain.actions
        chain_attributes = causal_chain.attributes

        # find indices where states, actions, and attributes of the chain match the causal relation
        # if any of them don't match, we can't disprove this chain
        matching_indices = []
        for i in range(num_states):
            if (
                chain_states[i] == causal_relation_state
                and chain_actions[i] == causal_relation_action
                and chain_attributes[i] == causal_relation_attributes
            ):
                matching_indices.append(i)
        # matching_indices = [
        #     i
        #     for i in range(num_states)
        #     if chain_states[i] == causal_relation_state
        #     and chain_actions[i] == causal_relation_action
        #     and chain_attributes[i] == causal_relation_attributes
        # ]

        # check causal relation types match for all matching schema indices
        for i in range(len(matching_indices)):
            schema_idx = matching_indices[i]
            schema_state = "state" + str(schema_idx)
            chain_causal_relation_type = self.extract_causal_relation_from_chain(
                causal_chain, schema_state
            )
            if chain_causal_relation_type != causal_relation_type:
                return False
        return True

    def get_all_chains_with_actions(self, actions):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_actions = self.get_actions(causal_chain_idx)
            if self.equals(chain_actions, actions):
                chains.append(causal_chain_idx)
        return chains

    def get_all_chains_with_attributes(self, attributes):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_attributes = self.get_attributes(causal_chain_idx)
            if self.equals(chain_attributes, attributes):
                chains.append(causal_chain_idx)
        return chains

    def pretty_print_causal_chain_idxs(
        self,
        causal_chain_idxs,
        beliefs=None,
        energies=None,
        q_values=None,
        belief_label="belief",
        print_messages=True
    ):
        # suppress printing
        if not print_messages:
            return

        table = texttable.Texttable()

        chain_content = []
        if len(causal_chain_idxs) > 0:
            for i in range(len(causal_chain_idxs)):
                new_chain_content = self.pretty_list_compact_chain(causal_chain_idxs[i])
                if beliefs is not None:
                    new_chain_content.append(beliefs[causal_chain_idxs[i]])
                if q_values is not None:
                    new_chain_content.append(q_values[causal_chain_idxs[i]])
                if energies is not None:
                    new_chain_content.append(energies[causal_chain_idxs[i]])
                chain_content.append(new_chain_content)
        else:
            return

        num_subchains = self.num_subchains_in_chain
        headers = ["idx"]
        headers.extend(["subchain{}".format(i) for i in range(num_subchains)])

        alignment = ["l"]
        alignment.extend(["l" for i in range(num_subchains)])

        widths = [5]
        widths.extend([60 for i in range(num_subchains)])

        if beliefs is not None:
            headers.append(belief_label)
            alignment.append("l")
            widths.append(15)
        if q_values is not None:
            headers.append("q-value")
            alignment.append("l")
            widths.append(15)
        if energies is not None:
            headers.append("energy")
            alignment.append("l")
            widths.append(15)

        content = [headers]
        content.extend(chain_content)
        table.add_rows(content)
        table.set_cols_align(alignment)
        table.set_cols_width(widths)
        print(table.draw())

    def pretty_list_compact_chain(self, causal_chain):
        # argument causal_chain is an index into self.causal_chains
        if isinstance(causal_chain, int) or np.issubdtype(causal_chain, np.integer):
            chain_chain = self.causal_chains[causal_chain]
        else:
            raise ValueError(
                "Expected causal chain index as causal_chain argument to pretty_list_compact_chain()"
            )
        l = [causal_chain]
        l.extend(
            [
                self.pretty_str_causal_relation(causal_relation)
                for causal_relation in chain_chain
            ]
        )
        return l

    @staticmethod
    def pretty_str_causal_relation(causal_relation):
        return "pre={},action={},attr={},fluent_change={}".format(
            causal_relation.precondition,
            causal_relation.action,
            causal_relation.attributes,
            causal_relation.causal_relation_type,
        )

    def print_random_set_of_causal_chains(self, causal_chain_idxs, num_chains=100):
        idxs = np.random.randint(0, len(causal_chain_idxs), size=num_chains)
        causal_chains_to_print = []
        for idx in idxs:
            causal_chains_to_print.append(causal_chain_idxs[idx])
        self.pretty_print_causal_chain_idxs(causal_chains_to_print, print_messages=self.print_messages)

    def pretty_print_causal_observations(self, causal_observations, print_messages=True):
        # suppress printing
        if not print_messages:
            return

        table = texttable.Texttable()
        table.set_cols_align(["l", "l"])
        content = [["step", "causal relation"]]
        for i in range(len(causal_observations)):
            causal_observation = causal_observations[i]
            content.append(
                [
                    str(i),
                    self.pretty_str_causal_relation(causal_observation.causal_relation),
                ]
            )
        table.add_rows(content)
        table.set_cols_width([7, 130])
        print(table.draw())

    def print_chains_above_threshold(self, belief_space, threshold):
        chains = []
        for causal_chain_idx in range(len(self.causal_chains)):
            chain_belief = belief_space.beliefs[causal_chain_idx]
            if chain_belief > threshold:
                chains.append(causal_chain_idx)
        if len(chains) > 0:
            self.pretty_print_causal_chain_idxs(chains, belief_space, print_messages=self.print_messages)

    @staticmethod
    def check_for_equality_or_containment(item, target):
        return item == target or item in target

    @staticmethod
    def extract_states_from_actions(actions):
        """
        extracts the states used in an action
        :param actions: a list of actions
        :return: states, the states used in actions
        """
        states = []
        # convert ActionLogs to str if needed
        if isinstance(actions[0], ActionLog):
            actions = [action.name for action in actions]
        for action in actions:
            state = action.split("_", 1)[1]
            assert state not in states, "Solutions should only use each lever/door once"
            states.append(state)
        return tuple(states)

    def write_to_json(self, filename):
        json_encoding_str = jsonpickle.encode(self)
        pretty_write(jsonpickle, filename)
