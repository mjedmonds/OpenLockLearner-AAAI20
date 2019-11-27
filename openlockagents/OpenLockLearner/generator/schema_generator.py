import itertools
import os
import re
import sys

import more_itertools
import networkx as nx

from openlockagents.OpenLockLearner.util.common import create_birdirectional_dict, ACTION_REGEX_STR, STATE_REGEX_STR

NO_CONSTRAINT = -1
UNASSIGNED_CHAIN = -1


def convert_chains_to_edges(chains):
    return list(set([tuple(edge) for chain in chains for edge in chain]))


class SchemaNodeID:
    """
    Dummy class to force all node IDs to be objects; allows using the same object in multiple chains
    Simplifies reassignment of nodes - node object is reassigned in one chain and change propagates to all chains
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if other is None:
            return False
        return self.value == other.value

CC_EDGES = ((SchemaNodeID(0),SchemaNodeID(1)), (SchemaNodeID(0),SchemaNodeID(2)))
CE_EDGES = ((SchemaNodeID(1),SchemaNodeID(0)), (SchemaNodeID(2),SchemaNodeID(0)))
CH_EDGES = ((SchemaNodeID(0),SchemaNodeID(1)), (SchemaNodeID(1),SchemaNodeID(2)))


def generate_atomic_schema_graphs():
    cc_graph = nx.DiGraph(CC_EDGES)
    ce_graph = nx.DiGraph(CE_EDGES)
    # ch_graph = nx.DiGraph(CH_EDGES)
    # return cc_graph, ce_graph, ch_graph
    return cc_graph, ce_graph


def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def generate_instantiation_mappings(n_chains_in_schema, solutions_executed):
    # for each schema, a valid mapping consists of every combination of solutions executed at each chain index

    # determine all the permutations of indices in the abstract schema
    mappings = list(
        itertools.permutations(range(n_chains_in_schema), n_chains_in_schema)
    )
    # all indices above len(solutions_executed are "free variables" and will not be instantiated, mark them as -1 and reduce to unique
    mappings = [
        tuple(
            ele if ele < len(solutions_executed) else UNASSIGNED_CHAIN
            for ele in mapping
        )
        for mapping in mappings
    ]
    mappings = list(set(mappings))

    return mappings



def generate_schemas(structure, n_paths, draw_chains=False):
    """
    generate all possible schema_chains with n_paths (solutions) using structure as the base
    :param structure:
    :param n_paths:
    :return:
    """
    base_state_chain, max_id_in_chain = extract_nodes_from_structure(structure)
    num_edges_in_chain = max_id_in_chain
    num_nodes_in_chain = num_edges_in_chain + 1
    node_list = list(itertools.chain(*base_state_chain))
    node_list = list(more_itertools.unique_everseen(node_list))
    base_node_id_list = list(range(max_id_in_chain + 1))

    # num_nodes = n_paths * (n_nodes_in_structure - 1) + 1
    num_nodes_needed = n_paths * max_id_in_chain + 1
    num_schemas_possibe = pow(2, num_nodes_needed - max_id_in_chain)

    # add to our node list
    for i in range(max_id_in_chain + 1, num_nodes_needed):
        node_list.append("state" + str(i))
        # node_list.append("action" + str(i))
    node_to_id_bidir_dict = create_birdirectional_dict(node_list, SchemaNodeID)

    # generate all possible chains from nodes
    node_id_list = [node_to_id_bidir_dict[x] for x in node_list]
    base_state_chain_id = [
        (node_to_id_bidir_dict[x], node_to_id_bidir_dict[y])
        for x, y in base_state_chain
    ]

    possible_edges = set(itertools.permutations(node_id_list, r=2))

    possible_chains = set(itertools.permutations(possible_edges, r=num_edges_in_chain))

    # remove base base chain (having the base chain produces duplicates)
    possible_chains.remove(tuple(base_state_chain_id))

    # check that each chain forms a valid chain (each child of an edge has a matching parent of the next edge
    possible_chains = [chain for chain in possible_chains if valid_chain(chain)]

    possible_chains_with_npaths = []
    for nchains in range(1, n_paths):
        possible_chains_with_npaths.extend(
            list(itertools.combinations(possible_chains, r=nchains))
        )

    counter = 0
    print_update_rate = 10000

    schema_chains = []
    schema_graphs = []
    chains_producing_schemas = []
    for chains in possible_chains_with_npaths:
        counter += 1
        if counter % print_update_rate == 0:
            print(
                "Generated {}/{} schema_chains with {} chains and {} nodes".format(
                    counter,
                    len(possible_chains_with_npaths),
                    n_paths,
                    num_nodes_in_chain,
                )
            )

        # create new chain from chain
        chains = list(chains)
        new_edges = set(itertools.chain(*chains))
        chains.append(tuple(base_state_chain_id))

        # add in base chain (required for all schema_chains
        new_edges.update(base_state_chain_id)
        new_edges = list(new_edges)

        # is_target_chain, target_chain_name = check_target(new_edges)
        # if is_target_chain:
        #     print("target schema: {}".format(target_chain_name))

        new_node_list = list(itertools.chain(*new_edges))
        new_node_list = set(new_node_list)

        # checks if this schema has a valid node list
        if not valid_node_ids(new_node_list):
            # if is_target_chain:
            #     print("removing target chain {}".format(target_chain_name))
            continue

        root_nodes = find_root_nodes(new_edges, new_node_list)
        leaf_nodes = find_leaf_nodes(new_edges, new_node_list)

        new_graph = nx.DiGraph(new_edges)

        # verify there are no disjoint subchains
        if disjoint_subchains_present(new_graph):
            # if is_target_chain:
            #     print("removing target chain {}".format(target_chain_name))
            continue

        # verify this chain has no cycles
        if len(list(nx.simple_cycles(new_graph))) != 0:
            # if is_target_chain:
            #     print("removing target chain {}".format(target_chain_name))
            continue

        # verify this chain has valid depths (specifically, every path from root to leaf
        # should have length num_nodes_in_chain)
        if not schema_depths_valid(
            new_edges, new_graph, root_nodes, leaf_nodes, num_nodes_in_chain, n_paths
        ):
            # if is_target_chain:
            #     print("removing target chain {}".format(target_chain_name))
            # schema_depths_valid(
            #     new_edges, new_graph, root_nodes, leaf_nodes, num_nodes_in_chain, n_paths
            # )
            continue

        # store schema as list of edges
        # schema_chains.append(new_edges)
        # store schema as list of chains
        schema_chains.append(chains)
        schema_graphs.append(new_graph)
        chains_producing_schemas.append(chains)

    # get set of schema edges of unique chains - by checking if each chain is isomporphic
    final_schema_chains = set()
    final_schema_graphs = set()
    for i in range(len(schema_graphs)):
        graph = schema_graphs[i]
        chain = tuple(schema_chains[i])
        # if chain1 is not in out schema set and it is not isomorphic to any chain in our schema set, add the chain to our schema set
        if chain not in final_schema_chains and all(
            [not nx.is_isomorphic(graph, final_graph) for final_graph in final_schema_graphs]
        ):
            final_schema_chains.add(chain)
            final_schema_graphs.add(graph)
    schema_chains = list(final_schema_chains)

    if draw_chains:
        save_chains_to_files(
            schema_chains,
            os.path.expanduser("~/Desktop/schema_chains/"),
            num_nodes_in_chain,
            n_paths,
        )

    print(
        "Generated {}/{} schema_chains with {} chains and {} nodes".format(
            len(possible_chains_with_npaths),
            len(possible_chains_with_npaths),
            n_paths,
            num_nodes_in_chain,
        )
    )
    return schema_chains


def check_target(new_edges):
    target_schema_edges = {
        "CE4": {(0, 1), (1, 2), (3, 1), (4, 1)},  # CE4
        "CC4": {(0, 1), (1, 2), (1, 3), (1, 4)},  # CC4
        "t1": {(0, 1), (1, 2), (0, 3), (0, 4), (3, 2), (4, 2)},
        "t2": {(0, 1), (1, 2), (0, 3), (0, 5), (3, 4), (5, 6)},
        "t3": {(0, 1), (1, 2), (3, 4), (5, 6), (4, 2), (6, 2)},
    }
    if set(new_edges) in target_schema_edges.values():
        return (
            True,
            list(target_schema_edges.keys())[
                list(target_schema_edges.values()).index(set(new_edges))
            ],
        )
    else:
        return False, None


def extract_nodes_from_structure(structure):
    def extract_number_from_node_name(node_name, regex):
        result = regex.search(node_name)
        if result is not None:
            return int(result.group(1))
        else:
            return None

    def determine_new_max(node_name, max_id_num, state_regex, action_regex):
        id_num = extract_number_from_node_name(node_name, state_regex)
        if id_num is not None and id_num > max_id_num:
            max_id_num = id_num
        id_num = extract_number_from_node_name(node_name, action_regex)
        if id_num is not None and id_num > max_id_num:
            max_id_num = id_num
        return max_id_num

    action_regex = re.compile(ACTION_REGEX_STR)
    state_regex = re.compile(STATE_REGEX_STR)

    max_id_num = -1
    state_nodes = list()
    for head, tail in structure:
        max_id_num = determine_new_max(head, max_id_num, state_regex, action_regex)
        max_id_num = determine_new_max(tail, max_id_num, state_regex, action_regex)
        # extract structure only including state nodes
        if re.match(state_regex, head) and re.match(state_regex, tail):
            if (head, tail) not in state_nodes:
                state_nodes.append((head, tail))
    return state_nodes, max_id_num


def valid_chain(chain):
    """
    Verifies chain is actually a sequentially connected chain
    :param chain: list of tuples representing a possible chain
    :return: boolean to indicate whether or not the chain forms a valid chain
    """

    for i in range(len(chain) - 1):
        parent_edge = chain[i]
        child_edge = chain[i + 1]
        # verify that the child of the parent edge (second node) matches the parent of the child edge (first node)
        if not parent_edge[1] == child_edge[0]:
            # if this isn't
            return False
    return True


def valid_node_ids(node_list):
    max_node_id = max(x.value for x in node_list)
    return max_node_id == (len(node_list) - 1)


def check_duplicate_schema(schema_edges, node_list):
    """
    determines if chain is a duplicate schema using the max node ID. For instance [(0,1),(1,2),(1,3)] would have a duplicate of [(0,1),(1,2),(1,4)].
    Structurally these are equivalent, but we may need to use the node 4 to describe more complex structures (hence why the duplicate chain is generated
    :param schema_edges: a potentially duplicate schema
    :return: boolean to indicate whether or not this chain is a duplicate of another chain with lower IDs
    """
    # checking for duplicates amounts to checking the min difference between any two nodes being greater than 1
    differences = []
    for node in node_list:
        min_diff = sys.maxsize
        for node2 in node_list:
            if node == node2:
                continue
            diff = abs(node - node2)
            if diff < min_diff:
                min_diff = diff
        differences.append(min_diff)
    # all differences should be 1 if this chain is not a duplicate
    return any(difference != 1 for difference in differences)


def schema_depths_valid(
    schema_edges, schema_chain, root_nodes, leaf_nodes, max_depth, n_paths
):
    """
    determines if schema_edges has depth less than or equal to max_depth
    :param schema_edges: edges to check
    :return: boolean to indicate whether or not schema_edges has depth greater than max_depth
    """
    max_chain_depth = -1
    min_chain_depth = sys.maxsize
    n_paths_chain = 0
    # find all paths between all root nodes and all leaf nodes, find the max length
    for root in root_nodes:
        n_paths_root = 0

        max_root_depth = -1
        min_root_depth = sys.maxsize

        for leaf in leaf_nodes:
            paths = list(nx.algorithms.all_simple_paths(schema_chain, root, leaf))

            # there must be at least one path between root and leaf to check depth
            if len(paths) == 0:
                continue

            max_leaf_depth = max(len(path) for path in paths)
            min_leaf_depth = min(len(path) for path in paths)
            if max_leaf_depth > max_root_depth:
                max_root_depth = max_leaf_depth
            if min_leaf_depth < min_root_depth:
                min_root_depth = min_leaf_depth

            n_paths_root += len(paths)

        n_paths_chain += n_paths_root

        if max_root_depth > max_chain_depth:
            max_chain_depth = max_root_depth
        if min_root_depth < min_chain_depth:
            min_chain_depth = min_root_depth

    # should have at least n_paths for the chain
    if n_paths_chain < n_paths:
        return False

    # both the min and the max path between any root and any leaf should be the length of max_depth
    # each path should complete an attempt
    return max_chain_depth == max_depth and min_chain_depth == max_depth


def disjoint_subchains_present(schema_chain):
    # convert to undirected
    undirected_schema_chain = schema_chain.to_undirected()
    subchains = list(nx.connected_component_subgraphs(undirected_schema_chain))
    # should only be one subchain, if there is more than 1, we have a disjoint subchain
    return len(subchains) != 1


def find_root_nodes(schema_edges, node_list):
    is_root_node = dict()
    for node in node_list:
        is_root_node[node] = True
    for head, tail in schema_edges:
        is_root_node[tail] = False
    return [node for node in node_list if is_root_node[node] is True]


def find_leaf_nodes(schema_edges, node_list):
    is_leaf_node = dict()
    for node in node_list:
        is_leaf_node[node] = True
    for head, tail in schema_edges:
        is_leaf_node[head] = False
    return [node for node in node_list if is_leaf_node[node] is True]


def save_chains_to_files(schemas, data_dir, num_nodes_in_chain, n_paths):
    c = 0
    for schema_chain in schemas:
        schema_edges = convert_chains_to_edges(schema_chain)
        chain = nx.DiGraph(schema_edges)
        base_filename = data_dir + "{}nodes_{}paths_{}".format(
            num_nodes_in_chain, n_paths, c
        )
        write_chain_to_file(chain, base_filename)
        c += 1


def write_chain_to_file(chain, filename):
    p = nx.drawing.nx_pydot.to_pydot(chain)
    p.write_png(os.path.expanduser(filename + ".png"))
    nx.drawing.nx_pydot.write_dot(chain, filename + ".dot")
