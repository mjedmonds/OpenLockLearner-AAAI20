import re

import openlockagents.OpenLockLearner.util.tsort as tsort
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import CausalRelationType
from openlockagents.OpenLockLearner.util.common import (
    GRAPH_INT_TYPE,
    STATE_REGEX_STR,
    ACTION_REGEX_STR,
)

class AndNode:
    def __init__(self):
        pass


class OrNode:
    def __init__(self):
        pass


class Node:
    def __init__(self, id, parents=None, children=None, instantiate_schema_ids=False):
        if parents is None:
            parents = []
        if children is None:
            children = []

        self.id = id
        # schema_id is to used to identify the id of this node in the schema
        self.schema_id = id if instantiate_schema_ids else None
        # depth of this node. Defined as the minimum depth of this node's parents
        self.depth = 0
        self.depths = []
        # indicates action or state
        self.type = None

        self.conditional_probability_table = None
        self.conditional_probability_table_labels = None
        # selects which row in the conditional probability table is active
        # in the discrete case, the CPT is a one-hot vector, only one causal encoding will be active at a time
        self.active_conditional_probability_table_row = None

        self.parents = parents
        self.children = children

    def convert_state_change_to_causal_relation(self, previous_state, current_state):
        # 0 -> 0
        if previous_state == False and current_state == False:
            return CausalRelationType.zero_to_zero
        # 0 -> 1
        if previous_state == False and current_state == True:
            return CausalRelationType.zero_to_one
        # 1 -> 0
        if previous_state == True and current_state == False:
            return CausalRelationType.one_to_zero
        # 1 -> 1
        if previous_state == True and current_state == True:
            return CausalRelationType.one_to_one

    # each CPT has the following form:
    # action_parent[0] action_parent[1] .... action_parent[n] state_parent[0] state_parent[1] .... state_parent[n] previous_state, current_state
    def convert_cpt_row_idx_to_causal_relation_type(self, cpt_row_idx):
        previous_state = self.conditional_probability_table[cpt_row_idx][-2]
        current_state = self.conditional_probability_table[cpt_row_idx][-1]
        return self.convert_state_change_to_causal_relation(
            previous_state, current_state
        )

    def convert_causal_relation_type_to_cpt_row_idx(self, causal_relation_type):
        for cpt_row_idx in range(len(self.conditional_probability_table)):
            previous_state = self.conditional_probability_table[cpt_row_idx][-2]
            current_state = self.conditional_probability_table[cpt_row_idx][-1]
            cpt_relation = self.convert_state_change_to_causal_relation(
                previous_state, current_state
            )
            if cpt_relation == causal_relation_type:
                return cpt_row_idx

    def swap_ids(self, new_ids):
        def match_id(node_id):
            schema_form_regex = r"([a-z]+)([0-9]+)"
            match = re.match(schema_form_regex, node_id, re.I)
            if not match:
                raise ValueError("ID is not of regex form: [a-zA-Z]+[0-9]+")

            node_id, node_idx = match.groups()
            # will be the index into the new ids
            node_idx = int(node_idx)
            new_id = new_ids[node_id + "s"][node_idx]
            if self.conditional_probability_table_labels is not None:
                # replace name
                try:
                    self.conditional_probability_table_labels[
                        self.conditional_probability_table_labels.index(self.id)
                    ] = new_id
                except ValueError:
                    pass
                # replace name_prev
                try:
                    self.conditional_probability_table_labels[
                        self.conditional_probability_table_labels.index(
                            self.id + "_prev"
                        )
                    ] = (new_id + "_prev")
                except ValueError:
                    pass
                # replace name_cur
                try:
                    self.conditional_probability_table_labels[
                        self.conditional_probability_table_labels.index(
                            self.id + "_cur"
                        )
                    ] = (new_id + "_cur")
                except ValueError:
                    pass

            return new_id

        self.id = match_id(self.id)
        # change parents and children
        for i in range(len(self.parents)):
            self.parents[i] = match_id(self.parents[i])
        for i in range(len(self.children)):
            self.children[i] = match_id(self.children[i])

        return self.id
        # items is ("foo", "21")


class CausalChain:
    def __init__(self, node_id_edges=None, schema_chain=False):
        # node list by node id
        self.node_id_to_node_dict = dict()
        # adjacency matrix of chain
        # self.adjacency_matrx = np.zeros((len(node_space), len(node_space)))
        self.id = "schema" if schema_chain else None

        # add all edges to the chain
        for edge in node_id_edges:
            self.add_edge(edge[0], edge[1], schema_chain=schema_chain)

        # assert np.amax(self.adjacency_matrx) == 1, 'Same edge added more than once'

        self.is_dag = self.determine_directed_acyclical_chain()

        if self.is_dag:
            self.root_nodes = self.find_root_nodes()
            self.compute_depths()
            self.depth = self.compute_chain_depth()
            self.width = self.compute_chain_width()

    @property
    def num_states_in_chain(self):
        return len(
            [
                x
                for x in self.node_id_to_node_dict.keys()
                if re.match(STATE_REGEX_STR, x)
            ]
        )

    @property
    def num_actions_in_chain(self):
        return len(
            [
                x
                for x in self.node_id_to_node_dict.keys()
                if re.match(ACTION_REGEX_STR, x)
            ]
        )

    # converts to dictionary of nodes as keys and their corresponding children as children
    def convert_chain_to_dict(self):
        chain_dict = dict()
        for label, node in self.node_id_to_node_dict.items():
            child_list = []
            for child_id in node.children:
                child_list.append(child_id)
            if len(child_list) != 0:
                chain_dict[node.id] = child_list
        return chain_dict

    # adds a node with corresponding label
    def add_node(self, node_id, parents=None, children=None, schema_chain=False):
        new_node = Node(node_id, parents, children, instantiate_schema_ids=schema_chain)
        self.node_id_to_node_dict[node_id] = new_node

    # adds an edge from parent to child
    def add_edge(self, parent_id, child_id, schema_chain=False):
        if parent_id not in self.node_id_to_node_dict.keys():
            self.add_node(
                parent_id, parents=None, children=None, schema_chain=schema_chain
            )
        if child_id not in self.node_id_to_node_dict.keys():
            self.add_node(
                child_id, parents=None, children=None, schema_chain=schema_chain
            )

        self.node_id_to_node_dict[parent_id].children.append(child_id)
        self.node_id_to_node_dict[child_id].parents.append(parent_id)
        # self.adjacency_matrx[parent_id, child_id] += 1

    # find root nodes in chain
    def find_root_nodes(self):
        root_nodes = []
        for node_label in self.node_id_to_node_dict.keys():
            # root nodes have children but have no parents
            if (
                len(self.node_id_to_node_dict[node_label].children) != 0
                and len(self.node_id_to_node_dict[node_label].parents) == 0
            ):
                root_nodes.append(node_label)
        return root_nodes

    def compute_depths(self):
        # compute depths, root nodes have depth of 0
        for root_node_id in self.root_nodes:

            self.node_id_to_node_dict[root_node_id].depths.append(0)
            self.compute_depths_recursive(self.node_id_to_node_dict[root_node_id], 0)
        # cleanup and consolidate depths. final depth is max depth found
        for id, node in self.node_id_to_node_dict.items():
            if len(node.depths) == 0:
                node.depth = -1
            else:
                node.depth = max(node.depths)

    def compute_depths_recursive(self, node, depth):
        for child_id in node.children:
            self.node_id_to_node_dict[child_id].depths.append(depth + 1)
            self.compute_depths_recursive(
                self.node_id_to_node_dict[child_id], depth + 1
            )

    # determines if this chain is a directed acyclical chain
    # only checks for cycles, we know this chain is directed
    def determine_directed_acyclical_chain(self):
        try:
            tsort.topological_sort(self.convert_chain_to_dict())
            return True
        except ValueError:
            return False

    # assumes depths have been calculated and chain depth has been calculated
    def compute_chain_width(self):
        # create list of nodes at each depth
        depth_list = [None] * (self.depth + 1)
        # collect the number of nodes at each depth
        for node_id, node in self.node_id_to_node_dict.items():
            if node.depth < 0:
                continue
            if depth_list[node.depth] is None:
                depth_list[node.depth] = set()
            depth_list[node.depth].add(node_id)
        # width is the depth with the largest number of unique elements
        width = -1
        for depth_set in depth_list:
            if width < len(depth_set):
                width = len(depth_set)
        return width

    # assumes depths have been calculated
    def compute_chain_depth(self):
        max_depth = -1
        for node in self.node_id_to_node_dict.values():
            if max_depth < node.depth:
                max_depth = node.depth
        return max_depth

    # def get_chain_depth(self):
    #     path_matrix = self.adjacency_matrx
    #     if np.amax(path_matrix) == 0:
    #         return 0
    #     depth = 0
    #     while np.amax(path_matrix) > 0:
    #         multiplying the path matrix by itself determines the next iterations of possible steps
    # path_matrix = np.matmul(path_matrix, path_matrix)
    # depth += 1
    # return depth


# compact representation of chain. Assumes a corresponding schema exists to define structure/CPTS
# essentially provides a particular instantiation of a schema
class CausalChainCompact:
    def __init__(
        self,
        states,
        actions,
        conditional_probability_table_choices,
        outcomes,
        attributes=None,
        belief=None,
        belief_count=None,
        belief_subchain_counts=None,
    ):
        states = tuple(states)
        actions = tuple(actions)
        conditional_probability_table_choices = tuple(
            conditional_probability_table_choices
        )
        attributes = tuple(attributes)
        outcomes = tuple(outcomes)

        assert isinstance(states, tuple), "States are expected in a tuple"
        assert isinstance(actions, tuple), "Actions are expected in a tuple"
        if attributes is not None:
            assert isinstance(actions, tuple), "Attributes are expected in a tuple"
        assert_str = "Expected conditional_probability_table_choices as {} in a tuple".format(
            GRAPH_INT_TYPE.__name__
        )
        assert isinstance(conditional_probability_table_choices, tuple) and isinstance(
            conditional_probability_table_choices[0], GRAPH_INT_TYPE
        ), assert_str

        self.states = states
        self.actions = actions
        self.attributes = attributes
        self.outcomes = outcomes
        # indices corresponding to conditional probability tables stored in the schema
        self.conditional_probability_table_choices = (
            conditional_probability_table_choices
        )
        # index in full causal chain space
        # overall belief in this chain, must start as non-zero. Will be set to zero if chain is found to be inconsistent with data
        self.belief = belief if belief is not None else 1
        self.belief_count = belief_count if belief_count is not None else 1
        # belief counts per subchains. Indices are consistent with schema indexing from conditional_probability_table_choices
        self.belief_subchain_counts = (
            belief_subchain_counts
            if belief_subchain_counts is not None
            else [0] * len(self.conditional_probability_table_choices)
        )

    def __eq__(self, other):
        return (
            self.states == other.states
            and self.actions == other.actions
            and self.conditional_probability_table_choices
            == other.conditional_probability_table_choices
            and self.outcomes == other.outcomes
            and self.attributes == other.attributes
        )

    def __str__(self):
        return str(
            {
                "states": self.states,
                "actions": self.actions,
                "attributes": self.attributes,
                "cpt_choices": self.conditional_probability_table_choices,
                "belief_counts": self.belief_subchain_counts,
                "belief": self.belief,
            }
        )

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    # used to convert to list for printing
    def __iter__(self):
        yield self.states
        yield self.actions
        yield self.attributes
        yield self.conditional_probability_table_choices
        yield self.belief
        yield self.belief_subchain_counts

