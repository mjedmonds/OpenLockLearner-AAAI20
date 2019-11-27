import multiprocessing
import sys
import time
import numpy as np

from joblib import Parallel, delayed
import networkx as nx

from openlockagents.OpenLockLearner.util.common import (
    SANITY_CHECK_ELEMENT_LIMIT,
    generate_slicing_indices,
    PARALLEL_MAX_NBYTES,
    verify_valid_probability_distribution,
    renormalize,
)
from openlockagents.OpenLockLearner.generator.schema_generator import (
    generate_instantiation_mappings,
    UNASSIGNED_CHAIN,
)

from openlockagents.OpenLockLearner.causal_classes.BeliefSpace import (
    AtomicSchemaBeliefSpace,
    AbstractSchemaBeliefSpace,
    InstantiatedSchemaBeliefSpace,
    BottomUpChainBeliefSpace,
    TopDownChainBeliefSpace,
)
from openlockagents.OpenLockLearner.causal_classes.SchemaStructureSpace import (
    InstantiatedSchemaStructureSpace,
)


def update_bottom_up_beliefs_multiproc(
    top_down_bottom_up_structure_belief_space_wrapper, attribute_order, trial_name
):
    slicing_indices = generate_slicing_indices(
        top_down_bottom_up_structure_belief_space_wrapper.bottom_up_belief_space.beliefs
    )
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        return_tuples = parallel(
            delayed(
                top_down_bottom_up_structure_belief_space_wrapper.update_bottom_up_beliefs_common
            )(attribute_order, trial_name, slicing_indices[i - 1], slicing_indices[i])
            for i in range(1, len(slicing_indices))
        )
        for return_tuple in return_tuples:
            returned_beliefs, starting_index, ending_index = return_tuple
            # copy beliefs from results
            top_down_bottom_up_structure_belief_space_wrapper.bottom_up_belief_space.beliefs[
                starting_index:ending_index
            ] = returned_beliefs

        return (
            top_down_bottom_up_structure_belief_space_wrapper.bottom_up_belief_space.beliefs
        )


def instantiate_schemas_multiproc(
    abstract_schema_space,
    causal_chain_structure_space,
    instantiation_index_mappings,
    solutions_executed,
    excluded_chain_idxs,
    n_chains_in_schema,
):
    slicing_indices = generate_slicing_indices(abstract_schema_space.structure_space)
    instantiated_schemas = []
    instantiated_schema_beliefs = []
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        return_tuples = parallel(
            delayed(abstract_schema_space.instantiate_schemas_common)(
                causal_chain_structure_space,
                instantiation_index_mappings,
                solutions_executed,
                excluded_chain_idxs,
                n_chains_in_schema,
                slicing_indices[i - 1],
                slicing_indices[i],
            )
            for i in range(1, len(slicing_indices))
        )
        for return_tuple in return_tuples:
            returned_schemas, returned_beliefs, starting_index, ending_index = (
                return_tuple
            )
            # copy beliefs from results
            instantiated_schemas.extend(returned_schemas)
            instantiated_schema_beliefs.extend(returned_beliefs)

        return instantiated_schemas, instantiated_schema_beliefs


class Structures:
    # Todo: implement abstract wrapper for all structures (schemas, chains, etc)
    pass


class Beliefs:
    # Todo: implement abstract wrapper for all beliefs (schemas, chains, etc) - this is basically done in BeliefManager
    pass


class StructureAndBeliefSpaceWrapper:
    def __init__(self, structures, beliefs):
        self.structure_space = structures
        self.belief_space = beliefs

    def __getitem__(self, item):
        return self.structure_space[item], self.belief_space[item]


class AtomicSchemaStructureAndBeliefWrapper(StructureAndBeliefSpaceWrapper):
    def __init__(self, structures, beliefs):
        assert isinstance(
            beliefs, AtomicSchemaBeliefSpace
        ), "Beliefs expected to be AtomicSchemaBeliefSpace"
        super(AtomicSchemaStructureAndBeliefWrapper, self).__init__(structures, beliefs)

    def find_closest_schema_match(self, graph):
        """
        finds the closest atomic match to the structure of graph using the graph edit distance
        :param graph: graph to match
        :return: the atomic schema and the index of the atomic schema with minimal graph edit distance
        """
        min_atomic_schema = None
        min_atomic_belief = None
        min_atomic_index = None
        min_graph_edit_distance = sys.maxsize
        for i in range(len(self.structure_space)):
            atomic_schema_graph = self.structure_space[i]
            edit_distance = nx.algorithms.graph_edit_distance(
                atomic_schema_graph, graph
            )
            if edit_distance < min_graph_edit_distance:
                min_atomic_schema = atomic_schema_graph
                min_atomic_belief = self.belief_space[i]
                min_atomic_index = i
                min_graph_edit_distance = edit_distance
        return min_atomic_schema, min_atomic_belief, min_atomic_index

    def update_atomic_schema_beliefs(self, completed_solutions):
        solution_edges = list(
            set(
                [
                    (str(completed_solution[i]), str(completed_solution[i + 1]))
                    for completed_solution in completed_solutions
                    for i in range(len(completed_solution) - 1)
                ]
            )
        )
        solution_graph = nx.DiGraph(solution_edges)

        # compute graph edit distance between solution and atomic schemas
        # update the beliefs of the atomic schema with the lowest graph edit distance to the solution schema
        min_atomic_schema, min_atmoic_belief, min_atomic_index = self.find_closest_schema_match(
            solution_graph
        )
        # update the dirichlet distribution of the min atomic schema
        self.belief_space.update_alpha(min_atomic_index)


class AbstractSchemaStructureAndBeliefWrapper(StructureAndBeliefSpaceWrapper):
    def __init__(self, structures, beliefs):
        assert isinstance(
            beliefs, AbstractSchemaBeliefSpace
        ), "Beliefs expected to be AbstractSchemaBeliefSpace"
        super(AbstractSchemaStructureAndBeliefWrapper, self).__init__(
            structures, beliefs
        )

    def instantiate_schemas(
        self,
        solutions_executed,
        n_chains_in_schema,
        causal_chain_structure_space,
        excluded_chain_idxs,
        multiproc=False,
    ):
        t = time.time()
        instantiated_schema_structure_space = InstantiatedSchemaStructureSpace()
        # mapping from solution_executed indices to chains in the schemas. -1 is a free index that will be filled with combinations
        instantiation_index_mappings = generate_instantiation_mappings(
            n_chains_in_schema, solutions_executed
        )

        if multiproc:
            instantiated_schemas, instantiated_schema_beliefs = instantiate_schemas_multiproc(
                abstract_schema_space=self,
                causal_chain_structure_space=causal_chain_structure_space,
                instantiation_index_mappings=instantiation_index_mappings,
                solutions_executed=solutions_executed,
                n_chains_in_schema=n_chains_in_schema,
                excluded_chain_idxs=excluded_chain_idxs,
            )
        else:
            instantiated_schemas, instantiated_schema_beliefs, _, _ = self.instantiate_schemas_common(
                causal_chain_structure_space=causal_chain_structure_space,
                instantiation_index_mappings=instantiation_index_mappings,
                solutions_executed=solutions_executed,
                n_chains_in_schema=n_chains_in_schema,
                excluded_chain_idxs=excluded_chain_idxs,
                starting_index=0,
                ending_index=len(self.structure_space),
            )

        instantiated_schema_structure_space.schemas = instantiated_schemas

        print(
            "Instantiating {0} schemas took {1:.2f}s".format(
                len(instantiated_schema_structure_space), time.time() - t
            )
        )
        instantiated_schema_beliefs = np.array(instantiated_schema_beliefs)

        # todo: does this renormalization make sense? Should we naturally have a normalized probability?
        instantiated_schema_beliefs = renormalize(instantiated_schema_beliefs)

        assert verify_valid_probability_distribution(
            instantiated_schema_beliefs
        ), "Instantiated Schema beliefs is not a valid probability distribution"

        return instantiated_schema_structure_space, instantiated_schema_beliefs

    def instantiate_schemas_common(
        self,
        causal_chain_structure_space,
        instantiation_index_mappings,
        solutions_executed,
        excluded_chain_idxs,
        n_chains_in_schema,
        starting_index,
        ending_index,
    ):
        instantiated_schemas = []
        instantiated_schema_beliefs = []
        total_num_instantiated_schemas = 0

        # for each abstract schema, instantiate in every possible valid way
        for i in range(starting_index, ending_index):
            abstract_schema = self.structure_space[i]
            abstract_schema_belief = self.belief_space[i]

            # todo: this is a hack to skip schemas that don't have a chain defined for every path possible
            # todo: this is a minor bug in schema generation - the graph adheres to constraints but we don't properly store the chains - none of these are solutions and therefore their dirichlets are never updated
            if len(abstract_schema.chains) != n_chains_in_schema:
                continue

            # create map of node indices in chain to assignments
            for index_mapping in instantiation_index_mappings:
                chain_assignments = [
                    solutions_executed[index_mapping[i]]
                    if index_mapping[i] >= 0
                    else UNASSIGNED_CHAIN
                    for i in range(len(index_mapping))
                ]

                exclusion_constraints = set(
                    [
                        relation
                        for solution_idx in solutions_executed
                        for relation in causal_chain_structure_space.causal_chains[
                            solution_idx
                        ]
                    ]
                )

                self.structure_space.generate_instantiated_schemas(
                    abstract_schema,
                    abstract_schema_belief,
                    chain_assignments,
                    causal_chain_structure_space,
                    instantiated_schemas,
                    exclusion_constraints,
                    excluded_chain_idxs,
                )

            num_instantiated_schemas = (
                len(instantiated_schemas) - total_num_instantiated_schemas
            )

            instantiated_schema_beliefs.extend(
                [abstract_schema_belief] * num_instantiated_schemas
            )
            total_num_instantiated_schemas = len(instantiated_schemas)

        return (
            instantiated_schemas,
            instantiated_schema_beliefs,
            starting_index,
            ending_index,
        )

    def update_abstract_schema_beliefs(self, atomic_schema_space, multiproc=False):
        # for every structure, find the closet matching atomic schema and adopt its belief
        for i in range(len(self.structure_space)):
            abstract_schema = self.structure_space[i]
            min_atomic_schema, min_atomic_belief, min_atomic_index = atomic_schema_space.find_closest_schema_match(
                abstract_schema.graph
            )
            self.belief_space[i] = min_atomic_belief

        # todo: we basically expand dimensions of a 2-value probability into n-value distribution - so we need to normalize
        # todo: if we had a proper conditional term, we could multiply it to get a valid distribution without normalization
        self.belief_space.renormalize_beliefs(multiproc=multiproc)

        assert verify_valid_probability_distribution(
            self.belief_space
        ), "AbstractSchemaSpace beliefs is not a valid probability distribution"

    # def update_abstract_schema_beliefs(self, completed_solutions):
    #     assert False, "Function deprecated"
    #     # construct solution chain using actions; we only care about the structure, so using actions is fine
    #     solution_edges = list(
    #         set(
    #             [
    #                 (str(completed_solution[i]), str(completed_solution[i + 1]))
    #                 for completed_solution in completed_solutions
    #                 for i in range(len(completed_solution) - 1)
    #             ]
    #         )
    #     )
    #     solution_graph = nx.DiGraph(solution_edges)
    #
    #     # find solution chain schema index by looking for isomorphic
    #     solution_idx = None
    #     solution_found = False
    #     for i in range(len(self.structure_space)):
    #         schema_edges = self.structure_space[i].edges
    #         schema_graph = nx.DiGraph(schema_edges)
    #         if nx.is_isomorphic(schema_graph, solution_graph):
    #             assert (
    #                 not solution_found
    #             ), "More than one schema is isomorphic to schema chain"
    #             solution_idx = i
    #             solution_found = True
    #
    #     assert solution_found, "Solutions executed could not be matched to abstract schema structure"
    #     # update distribution params
    #     self.belief_space.update_alpha(solution_idx)


class InstantiatedSchemaStructureAndBeliefWrapper(StructureAndBeliefSpaceWrapper):
    def __init__(self, structures, beliefs):
        assert isinstance(
            beliefs, InstantiatedSchemaBeliefSpace
        ), "Beliefs expected to be InstantiatedSchemaBeliefSpace"
        super(InstantiatedSchemaStructureAndBeliefWrapper, self).__init__(
            structures, beliefs
        )

    def update_instantiated_schema_beliefs(self, idxs_pruned, multiproc=False):
        idxs_pruned = set(idxs_pruned)
        for item_idx in range(len(self.belief_space)):
            schema_structure = self.structure_space[item_idx]
            # check for schemas to set zero belief to
            for chain_index in schema_structure.chain_idx_assignments:
                # if the chain was pruned, a chain in the instantiated schema is invalid - making the entire schema invalid
                if chain_index in idxs_pruned:
                    self.belief_space[item_idx] = 0
                    break
        # renormalize
        self.belief_space.renormalize_beliefs(multiproc=multiproc)


class TopDownBottomUpStructureAndBeliefSpaceWrapper:
    def __init__(self, structures, bottom_up_beliefs, top_down_beliefs):
        assert isinstance(
            bottom_up_beliefs, BottomUpChainBeliefSpace
        ), "Bottom up beliefs expected to be BottomUpChainBeliefSpace"
        assert isinstance(
            top_down_beliefs, TopDownChainBeliefSpace
        ), "Bottom up beliefs expected to be TopDownChainBeliefSpace"
        self.structure_space = structures
        self.bottom_up_belief_space = bottom_up_beliefs
        self.top_down_belief_space = top_down_beliefs

    def __getitem__(self, item):
        return (
            self.structure_space[item],
            self.bottom_up_belief_space[item],
            self.top_down_belief_space[item],
        )

    # updates beliefs based on counts stored with each chain
    def update_bottom_up_beliefs(self, attribute_order, trial_name, multiproc=False):

        if not multiproc:
            self.bottom_up_belief_space.beliefs, _, _ = self.update_bottom_up_beliefs_common(
                attribute_order,
                trial_name,
                starting_idx=0,
                ending_idx=len(self.bottom_up_belief_space),
            )
        else:
            self.bottom_up_belief_space.beliefs = update_bottom_up_beliefs_multiproc(
                self, attribute_order, trial_name
            )

        # renormalize beliefs
        max_chains = self.bottom_up_belief_space.renormalize_beliefs(
            multiproc=multiproc
        )

        # sanity checks in testing environments (small chain space)
        if (
            self.bottom_up_belief_space.num_idxs_with_belief_above_threshold
            < SANITY_CHECK_ELEMENT_LIMIT
        ):
            all_chain_idxs_have_belief_above_threshold = self.bottom_up_belief_space.verify_true_chain_idxs_have_belief_above_threshold(
                self.structure_space.true_chain_idxs
            )
            if not all_chain_idxs_have_belief_above_threshold:
                assert all_chain_idxs_have_belief_above_threshold
            num_idxs_correct = (
                self.bottom_up_belief_space.verify_num_idxs_with_belief_above_threshold_is_correct()
            )
            if not num_idxs_correct:
                assert num_idxs_correct

        return max_chains

    def update_bottom_up_beliefs_common(
        self, attribute_order, trial_name, starting_idx, ending_idx
    ):
        for causal_chain_idx in range(starting_idx, ending_idx):
            chain_belief = self.bottom_up_belief_space.beliefs[causal_chain_idx]

            # don't use belief_threshold here - we only want to eliminate chains we have completely disproven
            if chain_belief <= 0:
                continue

            chain_attributes = self.structure_space.get_attributes(causal_chain_idx)
            chain_actions = self.structure_space.get_actions(causal_chain_idx)

            # combine states with attributes (this is from before including position as an attribute
            # attributes_zipped = list(zip(causal_chain.states, causal_chain.attributes))

            # update belief using local attributes (both global and local attributes have already been updated from observation)
            new_belief = self.bottom_up_belief_space.attribute_space.local_attributes[
                trial_name
            ].compute_chain_posterior(
                attribute_order,
                chain_attributes,
                chain_actions,
                use_indexed_distributions=self.bottom_up_belief_space.use_indexed_distributions,
                use_action_distribution=self.bottom_up_belief_space.use_action_distribution,
            )
            self.bottom_up_belief_space.beliefs[causal_chain_idx] = new_belief
            if (
                new_belief == 0.0
                and causal_chain_idx in self.structure_space.true_chain_idxs
            ):
                new_belief = self.bottom_up_belief_space.attribute_space.local_attributes[
                    trial_name
                ].compute_chain_posterior(
                    attribute_order,
                    chain_attributes,
                    chain_actions,
                    use_indexed_distributions=self.bottom_up_belief_space.use_indexed_distributions,
                )
                raise RuntimeError(
                    "Problem, setting true chain to 0.0 belief through bottom-up process"
                )
            # old way - belief is this chain's belief count divided by normalization factor
            # causal_chain.belief = causal_chain.belief_count / normalization_factor

        return self.bottom_up_belief_space.beliefs[starting_idx:ending_idx], starting_idx, ending_idx

    def update_top_down_beliefs(self, instantiated_schema_space, multiproc=False):
        # mark all instantiated schemas that contain pruned causal chain idxs as invalid - 0 belief
        self.top_down_belief_space.set_zero_belief()
        for instantiated_schema_idx in range(
            len(instantiated_schema_space.structure_space)
        ):
            # recompute top-down belief in chains based on the schemas remaining for each chain
            if (
                instantiated_schema_space.belief_space[instantiated_schema_idx]
                > self.top_down_belief_space.belief_threshold
            ):
                instantiated_schema = instantiated_schema_space.structure_space[
                    instantiated_schema_idx
                ]
                for chain_index in instantiated_schema.chain_idx_assignments:
                    # every chain in the schema receives the belief in the schema - dervied from the belief in the abstract schema
                    self.top_down_belief_space[
                        chain_index
                    ] += instantiated_schema_space.belief_space[instantiated_schema_idx]

        # todo: does it make sense to renormalize here?
        self.top_down_belief_space.renormalize_beliefs(multiproc=multiproc)

        assert verify_valid_probability_distribution(
            self.top_down_belief_space
        ), "top_down_belief_space is not a valid probability distribution"
