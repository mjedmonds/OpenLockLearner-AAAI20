import copy
import time
from collections import defaultdict, namedtuple, OrderedDict
import texttable
import pickle as pkl

import numpy as np
import networkx as nx

from openlockagents.OpenLockLearner.generator.schema_generator import (
    generate_schemas,
    generate_atomic_schema_graphs,
    UNASSIGNED_CHAIN,
    NO_CONSTRAINT,
    convert_chains_to_edges,
)
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import CausalRelation


AbstractConstraint = namedtuple("AbstractConstraint", ["chain_index", "subchain_index"])
InstantiatedConstraint = namedtuple(
    "InstantiatedConstraint", ["causal_relation", "precondition_enforced"]
)
NodeEntry = namedtuple("NodeEntry", ["node_id", "chain_index"])


# helper to allow pickling
def defaultdict_set():
    return defaultdict(set)


class Schema:
    def __init__(self, chains, deep_init=False):
        # note: this deepcopy is very important. We want to isolate this Schema's SchemaNodeIDs, but we want
        # this schema to use the same object references for all SchemaNodeIDs. Deepcopying the chains and basing
        # all members off of the objects in the chains achieves this
        self.chains = copy.deepcopy(chains)
        self.graph = None
        self.root_nodes = None
        self.parent_map = None
        self.child_map = None
        self.constraints_between_chains = None
        if deep_init:
            self.deep_init()

    def deep_init(self):
        self.graph = nx.DiGraph(self.edges)
        self.root_nodes = self.generate_root_nodes(self.chains)
        self.parent_map, self.child_map = self.generate_parent_child_mapping(
            self.chains
        )
        self.constraints_between_chains = self.generate_constraints_between_chains()

    @property
    def edges(self):
        """
        returns the edges of this schema
        :return: a list of edges in this schema
        """
        return convert_chains_to_edges(self.chains)

    def remap_nodes(self, new_node_mapping):
        """
        Remaps the names of nodes to a new mapping
        :param new_node_mapping: dictionary mapping from old names to new names
        :return: None
        """
        # because all nodes are SchemaNodeIDs (i.e. objects), we only need to reassign nodes one way
        # changes propagate to chains, chain root_nodes, and parents automatically
        for chain in self.chains:
            for edge in chain:
                head, tail = edge
                if head in new_node_mapping.keys():
                    head.value = new_node_mapping[head]
                if tail in new_node_mapping.keys():
                    tail.value = new_node_mapping[tail]

    @staticmethod
    def generate_root_nodes(chains):
        return set([chain[0][0] for chain in chains])

    @staticmethod
    def generate_parent_child_mapping(chains):
        """
        Generate a mapping of parents to their children and children to their parents
        :param chains: a list of chains in the schema
        :return: two dictionaries of dictionaries of sets - e.g. for the parent_mapping, the outer dictionary indicates the starting node. The inner dictionary represents parents of the starting node. The set of the inner dictionary indicates the chains of that this parent is used in
        """

        parent_mapping = defaultdict(defaultdict_set)
        children_mapping = defaultdict(defaultdict_set)
        for chain_index in range(len(chains)):
            chain = chains[chain_index]
            for parent, child in chain:
                # add new parent to this child node's parent mapping
                parent_mapping[child][parent].add(chain_index)
                # add new child to this parent node's children mapping
                children_mapping[parent][child].add(chain_index)
                # ensure child node is also in child mapping, but add empty list to signify this child does have parents (yet)
        return parent_mapping, children_mapping

    def generate_constraints_between_chains(self):
        """
        generates constraints that chains could impose on one another.
        Any shared node between chains can act as a constraint when one chain is instantiated by the other is not.
        :return: a list of constraints per chain. Outer list is over chain indices. Inner lists are a list of tuples over subchain indices of this chain, where each tuple is (chain_index, subchain_index) where chain_index is the index of a chain that shares the node, and subchain_index is the subchain index at which that node exists in the other chain
        """
        node_to_chain_mapping = defaultdict(set)
        # collect all places where each node is used and at what subchain index
        for chain_idx in range(len(self.chains)):
            chain = self.chains[chain_idx]
            for subchain_idx in range(len(chain)):
                parent, child = chain[subchain_idx]
                node_to_chain_mapping[parent].add(
                    AbstractConstraint(chain_idx, subchain_idx)
                )
            # don't forget about the final child in the chain (parents are already accounted for)
            final_parent, final_child = chain[-1]
            node_to_chain_mapping[final_child].add(
                AbstractConstraint(chain_idx, len(chain))
            )
        # our final mapping correlates constraints on a per-chain basis
        # e.g. for chain index 0 at subchain index 1, we have a constraint (shared node) in chain 2
        chain_constraints = list()
        for chain_idx in range(len(self.chains)):
            chain = self.chains[chain_idx]
            chain_constraint = [set() for i in range(len(chain) + 1)]
            for subchain_idx in range(len(chain)):
                parent, child = chain[subchain_idx]
                node_constraints = node_to_chain_mapping[parent]
                for constraint in node_constraints:
                    if constraint.chain_index != chain_idx:
                        chain_constraint[subchain_idx].add(constraint)
            # don't forget about the final child in the chain (parents are already accounted for)
            final_parent, final_child = chain[-1]
            node_constraints = node_to_chain_mapping[final_child]
            for constraint in node_constraints:
                if constraint.chain_index != chain_idx:
                    chain_constraint[len(chain)].add(constraint)
            chain_constraints.append(chain_constraint)
        return chain_constraints


class InstantiatedSchema:
    def __init__(self, abstract_schema, causal_chain_idx_chain_assignments=None):
        if causal_chain_idx_chain_assignments is None:
            causal_chain_idx_chain_assignments = []
        self.abstract_schema = abstract_schema
        self.chain_idx_assignments = causal_chain_idx_chain_assignments


class SchemaStructureSpace:
    """
    Stub class for providing getitem for Instantiated and Abstract schema classes
    """

    def __init__(self):
        self.schemas = None

    def __getitem__(self, item):
        return self.schemas[item]

    def __len__(self):
        return len(self.schemas)


class InstantiatedSchemaStructureSpace(SchemaStructureSpace):
    def __init__(self, instantiated_schemas=None):
        super(SchemaStructureSpace, self).__init__()
        if instantiated_schemas is None:
            instantiated_schemas = []
        self.schemas = instantiated_schemas

    def reset(self):
        self.schemas = []

    def get_all_causal_chain_idxs_in_schemas(self):
        return [
            idx
            for schema in self.schemas
            for idx in schema.chain_idx_assignments
        ]

    def find_chain_idx_in_schemas(self, chain_idx):
        return [
            schema_idx
            for schema_idx in range(len(self.schemas))
            if chain_idx in self.schemas[schema_idx].chain_idx_assignments
        ]

    def find_chain_assignment_in_schemas(self, chain_assignment):
        for schema_idx in range(len(self.schemas)):
            schema = self.schemas[schema_idx]
            if schema.chain_idx_assignments == chain_assignment:
                return schema_idx
        return None

    def verify_chain_assignment_in_schemas(self, chain_assignment):
        return (
            True
            if self.find_chain_assignment_in_schemas(chain_assignment) is not None
            else False
        )


class AtomicSchemaStructureSpace(SchemaStructureSpace):
    def __init__(self):
        super(SchemaStructureSpace, self).__init__()
        self.schemas = generate_atomic_schema_graphs()

    def pretty_print(self, atomic_schema_belief_space=None, print_messages=True):
        if not print_messages:
            return

        table = texttable.Texttable()

        if atomic_schema_belief_space is not None:
            table_content = [
                [
                    schema_idx,
                    self.schemas[schema_idx].edges,
                    atomic_schema_belief_space.beliefs[schema_idx],
                    atomic_schema_belief_space.schema_dirichlet.frequency_count[schema_idx]
                ]
                for schema_idx in range(len(self.schemas))
            ]
            headers = ["schema_idx", "schema_edges", "belief", "alpha"]
            widths = [15, 100, 15, 15]
        else:
            table_content = [
                [schema_idx, self.schemas[schema_idx].edges]
                for schema_idx in range(len(self.schemas))
            ]
            headers = ["schema_idx", "schema_edges"]
            widths = [15, 100]

        alignment = ["c" for i in range(len(headers))]
        table.set_cols_align(alignment)
        content = [headers]
        content.extend(table_content)

        table.add_rows(content)

        table.set_cols_width(widths)

        print(table.draw())


class AbstractSchemaStructureSpace(SchemaStructureSpace):
    def __init__(self, structure, n_paths, draw_chains=False):
        super(SchemaStructureSpace, self).__init__()
        chains = generate_schemas(structure, n_paths, draw_chains)
        self.schemas = [Schema(chain, deep_init=True) for chain in chains]

    def __len__(self):
        return len(self.schemas)

    def generate_instantiated_schemas(
        self,
        abstract_schema,
        abstract_schema_belief,
        chain_assignments,
        causal_chain_structure_space,
        schemas,
        relations_used_in_instantiated_chains,
        excluded_chain_idxs,
    ):
        # all chains have been instantiated, create schema and add to instantiated schemas
        if UNASSIGNED_CHAIN not in chain_assignments:
            # causal_chain_space.pretty_print_causal_chains(chain_assignments)
            new_instantiated_schema = InstantiatedSchema(
                abstract_schema, chain_assignments
            )
            schemas.append(new_instantiated_schema)
        # we have at least one chain to still instantiate
        else:
            # pick first uninstantiated chain to instantiate
            chain_index = chain_assignments.index(UNASSIGNED_CHAIN)
            try:
                abstract_chain = abstract_schema.chains[chain_index]
            except IndexError:
                print("problem")
                raise IndexError("Invalid chain index in abstract schema")
            # determine all constraints on this chain from already-instantiated chains
            constraint_chains_by_subchain_index = abstract_schema.constraints_between_chains[
                chain_index
            ]
            # inclusion constraints come from instantiated chains with shared nodes - the instantiated nodes place constraints on what is permitted in this chain
            inclusion_constraints = [
                OrderedDict() for i in range(len(abstract_chain) + 1)
            ]
            # exclusion constraints start as all instantiated nodes - nodes will be removed as they are added as inclusion constraints
            exclusion_constraints = copy.copy(relations_used_in_instantiated_chains)
            prev_node_postcondition = None

            for subchain_idx in range(len(constraint_chains_by_subchain_index)):
                subchain_constraints = constraint_chains_by_subchain_index[subchain_idx]
                if len(subchain_constraints) > 0:
                    for subchain_constraint in subchain_constraints:
                        constraint_chain_index = subchain_constraint.chain_index
                        # if the chain index of this constraint is instantiated, we need to extract the constraints at this subchain index
                        if chain_assignments[constraint_chain_index] != NO_CONSTRAINT:
                            # get the node adding the constraints - from the causal chain index of the chain assignments at constraint's chain index.
                            # Then get the node at the subchain_index specified by the constraint
                            node_adding_constrant = causal_chain_structure_space.causal_chains[
                                chain_assignments[constraint_chain_index]
                            ][
                                subchain_constraint.subchain_index
                            ]

                            # if this node is in exclusion constraints, remove it. This node is part of an inclusion constraint
                            if node_adding_constrant in exclusion_constraints:
                                exclusion_constraints.remove(node_adding_constrant)

                            constraints = node_adding_constrant._asdict()

                            # no constraint coming from previous node
                            if prev_node_postcondition == NO_CONSTRAINT:
                                constraints.pop("precondition")
                            # copy the post condition from the previous instantiated node
                            else:
                                constraints["precondition"] = prev_node_postcondition

                            # if we have multiple constraints we must satisfy at this subchain, those constraints must match
                            # if the constraints do not match (if below), then this chain assignment is incompatible with this abstract schema.
                            # i.e. there is not possible way to assign these chains and form the given abstract schema
                            if len(inclusion_constraints[subchain_idx]) > 0 and inclusion_constraints[subchain_idx] != constraints:
                                # if we reach this point, there is no possible valid assignment using the provided chain_assignment.
                                # the constraints imposed by the existing chain_assignments are in conflict,
                                # there are not valid instantiated schemas down this portion of the search space
                                return
                                # try:
                                #     assert (
                                #         inclusion_constraints[subchain_idx] == constraints
                                #     ), "Constraint is already instantiated"
                                # except AssertionError:
                                #     print("problem")
                                #     raise AssertionError("Constraint is already instantiated")

                            inclusion_constraints[subchain_idx] = constraints
                # if we added constraints at this subchain, the postcondition of this
                if "attributes" in inclusion_constraints[subchain_idx].keys():
                    prev_node_postcondition = (
                        inclusion_constraints[subchain_idx]["attributes"],
                        inclusion_constraints[subchain_idx]["causal_relation_type"][1],
                    )
                # otherwise the next node has no precondition constraint
                else:
                    prev_node_postcondition = NO_CONSTRAINT

            # find all causal chain indices that adhere to constraints
            causal_chain_idxs_satisfying_constraints = causal_chain_structure_space.find_all_causal_chains_satisfying_constraints(
                inclusion_constraints,
                exclusion_constraints,
            )
            # remove excluded chain indices from the constraints (these chains have been pruned already)
            causal_chain_idxs_satisfying_constraints = (
                causal_chain_idxs_satisfying_constraints - excluded_chain_idxs
            )

            # for each causal chain index that adheres to constraints, recurse with new chain assignment
            for causal_chain_idx in causal_chain_idxs_satisfying_constraints:
                new_chain_assignments = copy.copy(chain_assignments)
                new_chain_assignments[chain_index] = causal_chain_idx
                # add relations from this causal chain index to exclusion constraints - only add constraints that aren't shared nodes
                new_relations_used_in_instantiated_chains = (
                    set()
                    if relations_used_in_instantiated_chains is None
                    else copy.copy(relations_used_in_instantiated_chains)
                )
                new_relations_used_in_instantiated_chains.update(
                    [
                        relation
                        for relation in causal_chain_structure_space.causal_chains[
                            causal_chain_idx
                        ]
                    ]
                )

                self.generate_instantiated_schemas(
                    abstract_schema,
                    abstract_schema_belief,
                    new_chain_assignments,
                    causal_chain_structure_space,
                    schemas,
                    new_relations_used_in_instantiated_chains,
                    excluded_chain_idxs,
                )

    @property
    def schemas_as_edge_list(self):
        """
        converts schemas stored as list of chains (each chain is a tuple of edges) to a list of edges
        :return: a list of schemas where each schema is represented as a list of edges
        """
        return [
            [tuple(edge) for chain in schema.chains for edge in chain]
            for schema in self.schemas
        ]

    @staticmethod
    def extract_nodes_in_chain(chain):
        node_order = []
        for parent, child in chain:
            node_order.append(parent)
        # append the last child
        node_order.append(chain[-1][1])
        return node_order

    @staticmethod
    def extract_nodes_in_chain_as_values(chain):
        node_order = []
        for parent, child in chain:
            node_order.append(parent.value)
        # append the last child
        node_order.append(chain[-1][1].value)
        return node_order

    @staticmethod
    def check_for_precondition_constraints(schema, node):
        # if all parents are CausalRelations, these parents add precondition constraints on the free node
        # if a node has no parents, the precondition is defined as None
        parents = schema.parent_map[node]
        # NO_CONSTRAINT indicates the precondition has no constraints
        precondition = NO_CONSTRAINT
        if len(parents) == 0:
            precondition = None
        elif len(parents) > 0:
            all_parents_instantiated = True
            for parent in parents:
                if not isinstance(parent.value, CausalRelation):
                    all_parents_instantiated = False
            # if all parents are instantiated, the precondition constraints are and OR of the parent's post-condition effects
            if all_parents_instantiated:
                parent_postconditions = [
                    (parent.value.attributes, parent.value.causal_relation_type[1])
                    for parent in parents
                ]
                assert (
                    len(parent_postconditions) < 2
                ), "multiple preconditions not implemented yet"
                precondition = (
                    tuple(parent_postconditions)
                    if len(parents) > 1
                    else parent_postconditions[0]
                )
        return precondition

    def pretty_print(self, abstract_schema_belief_space=None):
        table = texttable.Texttable()

        if abstract_schema_belief_space is not None:
            table_content = [
                [
                    schema_idx,
                    self.schemas[schema_idx].chains,
                    abstract_schema_belief_space.beliefs[schema_idx],
                ]
                for schema_idx in range(len(self.schemas))
            ]
            headers = ["schema_idx", "schema_chains", "belief"]
            widths = [15, 100, 15]
        else:
            table_content = [
                [schema_idx, self.schemas[schema_idx].chains]
                for schema_idx in range(len(self.schemas))
            ]
            headers = ["schema_idx", "schema_chains"]
            widths = [15, 100]

        alignment = ["c" for i in range(len(headers))]
        table.set_cols_align(alignment)
        content = [headers]
        content.extend(table_content)

        table.add_rows(content)

        table.set_cols_width(widths)

        print(table.draw())

    # @staticmethod
    # def check_for_postcondition_constraints(schema, node):
    #     """
    #     A post-condition constraint is enforced if this free node has one child, and that
    #     :param schema:
    #     :param node:
    #     :return:
    #     """
    #     children = schema.child_map[node]
    #     # collect the parents of every child
    #     child_parents = [schema.parent_map[n] for n in children]
    #     for child_parent in child_parents:
    #         # this
    #          if len(child_parent) > 1:
