import time
import multiprocessing
from joblib import Parallel, delayed

from openlockagents.OpenLockLearner.perceptual_causality_python.perceptual_causality import (
    load_perceptually_causal_relations,
)
from openlockagents.OpenLockLearner.util.common import (
    generate_slicing_indices,
    PARALLEL_MAX_NBYTES,
)

from openlockagents.common import DEBUGGING
from openlockagents.OpenLockLearner.util.common import print_message


def prune_chains_from_initial_observation_multiproc(
    chain_pruner, causal_chain_space, position_dict, trial_count, attempt_count
):
    slicing_indices = generate_slicing_indices(
        causal_chain_space.structure_space.causal_chains
    )
    chain_idxs_pruned = set()
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        chain_space_tuples = parallel(
            delayed(chain_pruner.prune_chains_from_initial_observation_common)(
                causal_chain_space,
                position_dict,
                trial_count,
                attempt_count,
                slicing_indices[i - 1],
                slicing_indices[i],
            )
            for i in range(1, len(slicing_indices))
        )
        for chain_space_tuple in chain_space_tuples:
            returned_beliefs, returned_chain_idxs_pruned, starting_index, ending_index = (
                chain_space_tuple
            )
            causal_chain_space.bottom_up_belief_space.beliefs[
                starting_index:ending_index
            ] = returned_beliefs
            chain_idxs_pruned.update(returned_chain_idxs_pruned)

        return causal_chain_space.bottom_up_belief_space.beliefs, chain_idxs_pruned


class ChainPruner:
    def __init__(self, print_messages):
        self.print_messages = print_messages

    def prune_chains_from_initial_observation(
        self,
        causal_chain_space,
        env,
        trial_count,
        attempt_count,
        multiproc=False,
        using_ids=True,
    ):
        """
        prunes chains based on the initial observation of the chains
        :param initial_observations: initial observations, specifically the attributes of every position
        :return: nothing
        """
        # todo: generalize this; it's manually defined for position
        position_to_color_dict = dict()
        attributes = [env.get_obj_attributes(obj) for obj in env.position_to_idx.keys()]
        if using_ids:
            for attributes_at_position in attributes:
                for attribute in attributes_at_position:
                    attributes_at_position[
                        attribute
                    ] = causal_chain_space.structure_space.unique_id_manager.convert_attribute_to_target_type(
                        attribute, attributes_at_position[attribute], target_type="int"
                    )

        for attribute in attributes:
            # assign the observed colors to each position
            position = attribute["position"]
            color = attribute["color"]
            assert (
                position not in position_to_color_dict.keys()
            ), "Same position seen twice!"
            position_to_color_dict[position] = color

        if multiproc:
            causal_chain_space.bottom_up_belief_space.beliefs, chain_idxs_pruned = prune_chains_from_initial_observation_multiproc(
                self,
                causal_chain_space,
                position_to_color_dict,
                trial_count,
                attempt_count,
            )
        else:
            causal_chain_space.bottom_up_belief_space.beliefs, chain_idxs_pruned, _, _ = self.prune_chains_from_initial_observation_common(
                causal_chain_space,
                position_to_color_dict,
                trial_count,
                attempt_count,
                0,
                len(causal_chain_space.structure_space.causal_chains),
            )

        causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold -= len(
            chain_idxs_pruned
        )

        return chain_idxs_pruned

    def prune_chains_from_initial_observation_common(
        self,
        causal_chain_space,
        position_to_color_dict,
        trial_count,
        attempt_count,
        starting_idx,
        ending_idx,
    ):
        position_index = causal_chain_space.structure_space.attribute_order.index(
            "position"
        )
        color_index = causal_chain_space.structure_space.attribute_order.index("color")
        chain_idxs_pruned = set()
        for causal_chain_idx in range(starting_idx, ending_idx):
            chain_chain = causal_chain_space.structure_space.causal_chains[
                causal_chain_idx
            ]
            for subchain in chain_chain:
                chain_attribute_position = subchain.attributes[position_index]
                chain_attribute_color = subchain.attributes[color_index]
                # if we have a mismatch, mark chain as invalid
                if (
                    chain_attribute_color
                    != position_to_color_dict[chain_attribute_position]
                ):
                    causal_chain_space.bottom_up_belief_space.beliefs[
                        causal_chain_idx
                    ] = 0.0
                    chain_idxs_pruned.add(causal_chain_idx)
                    break
                # check for invalid preconditions
                if subchain.precondition is not None:
                    chain_precondition_position = subchain.precondition[0][
                        position_index
                    ]
                    chain_precondition_color = subchain.precondition[0][color_index]
                    if (
                        chain_precondition_color
                        != position_to_color_dict[chain_precondition_position]
                    ):
                        causal_chain_space.bottom_up_belief_space.beliefs[
                            causal_chain_idx
                        ] = 0.0
                        chain_idxs_pruned.add(causal_chain_idx)
                        break

        print_message(
            trial_count,
            attempt_count,
            "Pruned {}/{} chains based on initial observation".format(
                len(chain_idxs_pruned), ending_idx - starting_idx
            ),
            self.print_messages
        )
        return (
            causal_chain_space.bottom_up_belief_space.beliefs[starting_idx:ending_idx],
            chain_idxs_pruned,
            starting_idx,
            ending_idx,
        )

    def prune_inconsistent_chains_v2(
        self, causal_chain_space, causal_chain_idxs, action_sequences_to_prune
    ):
        # todo: implement a more efficient version of this function using the action sequences that should be pruned, then find the indices of those action sequences and prune
        chain_idxs_removed_total = set()
        for pruned_seq in action_sequences_to_prune:
            chain_idxs_removed_total.update(
                causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                    pruned_seq
                )
            )
        # filter down to the causal chains remaining
        # todo: this intersection should work, but is not
        chain_idxs_removed = set(causal_chain_idxs).intersection(
            chain_idxs_removed_total
        )
        # chain_idxs_removed = chain_idxs_removed_total
        for chain_idx_to_prune in chain_idxs_removed:
            causal_chain_space.bottom_up_belief_space[chain_idx_to_prune] = 0
        chain_idxs_consistent = set(causal_chain_idxs) - chain_idxs_removed
        # sanity check, ALL of the chains that could be pruned should be - regardless of what chain idxs are in causal_chain_idxs. But causal_chain_idxs should reflect the causal chains that haven't been pruned, so if this assertion fails, causal_chain_idxs has a problem
        if DEBUGGING:
            assert all(
                [
                    causal_chain_space.bottom_up_belief_space[i] == 0
                    for i in chain_idxs_removed_total
                ]
            ), "causal_chain_idxs does not reflect all chains with positive belief. Should have removed chain that is not included in causal_chain_idxs"
        return chain_idxs_consistent, chain_idxs_removed

    def prune_inconsistent_chains(
        self,
        causal_chain_space,
        causal_chain_idxs,
        causal_observations,
        trial_count,
        attempt_count,
        multiproc=False,
    ):
        assert False, "function deprecated"
        # todo: this will only work in the deterministic case
        if causal_chain_space.structure_space.using_ids:
            causal_observations = [
                causal_chain_space.structure_space.unique_id_manager.convert_causal_observation_to_target_type(
                    x, target_type="int"
                )
                for x in causal_observations
            ]

        # if all([True if x.causal_relation_type is None else False for x in causal_observations ]):
        #     print('No effect')

        if not multiproc:
            chain_idxs_removed, chain_idxs_consistent = self.prune_inconsistent_chains_common(
                causal_chain_space,
                causal_chain_idxs,
                causal_observations,
                trial_count,
                attempt_count,
            )
        else:
            chain_idxs_removed, chain_idxs_consistent = self.prune_inconsistent_chains_common(
                causal_chain_space,
                causal_chain_idxs,
                causal_observations,
                trial_count,
                attempt_count,
            )
            # todo: cannot parallelize this; class functions are unpicklable
            # slicing_indices = generate_slicing_indices(self.causal_chain_space.structure_space.causal_chains)
            # with Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5) as parallel:
            #     chain_idxs_to_remove_and_chains_removed = parallel(delayed(self.prune_chains_with_inconsistent_conditional_probability_table_and_update_belief_counts_common)(self.causal_chain_space.structure_space.causal_chains[slicing_indices[i-1]:slicing_indices[i]], observed_state, observed_action, observed_causal_relation_type, observed_attributes) for i in range(len(slicing_indices)))
            #     chain_idxs_to_remove, chains_removed = zip(*chain_idxs_to_remove_and_chains_removed)

        # chains_pruned = prev_num_chains_with_positive_belief - self.causal_chain_space.structure_space.num_chains_with_positive_belief
        # assert len(chain_idxs_removed) == chains_pruned, "Number of chains removed doesn't match number change in number of chains with positive belief"

        return chain_idxs_consistent, chain_idxs_removed

    def prune_inconsistent_chains_common(
        self,
        causal_chain_space,
        causal_chain_idxs,
        causal_observations,
        trial_count,
        attempt_count,
    ):
        assert False, "function deprecated"
        chains_idxs_removed = []
        chains_idxs_consistent = []
        print_update_rate = 1000000
        start_time = time.time()

        # normalization_factor = 0
        for i in range(len(causal_chain_idxs)):
            if i % print_update_rate == 0 and i != 0:
                print_message(
                    trial_count,
                    attempt_count,
                    "Checking for chains to prune. {}/{} chains checked. Runtime: {:0.6f}s".format(
                        i, len(causal_chain_idxs), time.time() - start_time
                    ),
                    self.print_messages
                )

            # this represents where we are in the chain's transitions - not all observations are causal, so don't advance the chain's execution
            chain_change_idx = 0
            causal_chain_idx = causal_chain_idxs[i]
            chain = causal_chain_space.structure_space.causal_chains[causal_chain_idx]
            chain_belief = causal_chain_space.bottom_up_belief_space.beliefs[
                causal_chain_idx
            ]

            # target_states = ("UPPER", "door")
            # target_actions = ("push_UPPER", "push_door")
            # target_causal_relations = (
            #     CausalRelationType.one_to_zero,
            #     CausalRelationType.zero_to_one,
            # )
            # target_cpt_choices = self.causal_chain_space.structure_space.convert_causal_relations_to_cpt_choices(
            #     target_causal_relations
            # )
            # target_attributes = (("UPPER", "GREY"), ("door", "GREY"))
            # target_chain = self.causal_chain_space.structure_space.create_compact_chain(
            #     target_states,
            #     target_actions,
            #     target_cpt_choices,
            #     target_attributes,
            #     convert_to_ids=True,
            # )
            # if causal_chain.states == target_chain.states and causal_chain.actions == target_chain.actions:
            #     self.causal_chain_space.structure_space.pretty_print_causal_chains([causal_chain])
            # if causal_chain == target_chain:
            #     self.causal_chain_space.structure_space.pretty_print_causal_chains([causal_chain])
            # if causal_chain.actions == target_actions:
            #     self.causal_chain_space.structure_space.pretty_print_causal_chains([causal_chain])

            # if causal_chain in self.causal_chain_space.structure_space.true_chains:
            #     print('True chain found')

            # skip checking this chain if we have already pruned it (chains below threshold should still be considered)
            if chain_belief <= 0.0:
                continue

            chain_consistent = True
            for causal_observation in causal_observations:
                skipped_relation_flag, skip_chain_flag, outcome_consistent = self.check_node_consistency(
                    causal_chain_space=causal_chain_space,
                    causal_observation=causal_observation,
                    chain=chain,
                    chain_change_idx=chain_change_idx,
                )
                # received signal to skip the rest of this chain
                if skip_chain_flag:
                    break
                # received signal to skip relation
                if skipped_relation_flag:
                    continue
                # if the outcome is inconsistent on the first action, prune
                if not outcome_consistent:
                    # print("GRAPH {} PRUNED because of subchain {}:".format(causal_chain_idx, chain_change_idx))
                    # causal_chain_space.structure_space.pretty_print_causal_chains([causal_chain_idx])
                    assert (
                        chain_belief > 0.0
                    ), "Removing chain already marked as invalid"

                    causal_chain_space.bottom_up_belief_space[causal_chain_idx] = 0.0
                    # self.qlearner.remove_causal_chain_from_local_Q(trial_name, causal_chain.chain_id)

                    chains_idxs_removed.append(causal_chain_idx)
                    chain_consistent = False
                    if DEBUGGING:
                        if (
                            causal_chain_idx
                            in causal_chain_space.structure_space.true_chain_idxs
                        ):
                            print_message(trial_count, attempt_count, "GRAPH REMOVED: ")
                            causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
                                [causal_chain_idx]
                            )
                            raise ValueError("true chain removed from plausible chains")
                    # exit loop over observations, we are finished with this chain
                    break

                # advance a chain change index if there was a causal change (causal relation present).
                if causal_observation.determine_causal_change_occurred():
                    chain_change_idx += 1

            if chain_consistent:
                # print("GRAPH {} CONSISTENT through subchain {}:".format(causal_chain_idx, chain_change_idx))
                # causal_chain_space.structure_space.pretty_print_causal_chains([causal_chain_idx])
                chains_idxs_consistent.append(causal_chain_idx)

        return chains_idxs_removed, chains_idxs_consistent

    def check_node_consistency(
        self, causal_chain_space, causal_observation, chain, chain_change_idx
    ):
        # 1. we need to skip this causal relation if the causal relation's actions and preconditions do not match the chain's actions and preconditions AND the observed causal relation induced no state change
        # 2. we need to skip this chain if the chain's actions and preconditions do not match the causal relation's actions and preconditions AND the observed causal relation induced a state change
        skipped_relation_flag = False
        skip_chain_flag = False
        outcome_consistent = False
        if (
            causal_observation.causal_relation.action != chain[chain_change_idx].action
            or causal_observation.causal_relation.precondition
            != chain[chain_change_idx].precondition
        ):
            # 1. we need to skip this causal relation if the causal relation's actions and preconditions do not match the chain's actions and preconditions AND the observed causal relation induced no state change
            if not causal_observation.determine_causal_change_occurred():
                skipped_relation_flag = True
                return skipped_relation_flag, skip_chain_flag, outcome_consistent
            # 2. we need to skip this chain if the chain's actions and preconditions do not match the causal relation's actions and preconditions AND the observed causal relation induced a state change
            else:
                skip_chain_flag = True
                return skipped_relation_flag, skip_chain_flag, outcome_consistent

        outcome_consistent = self.check_outcome_consistency(
            causal_chain_space.structure_space,
            chain,
            causal_observation,
            chain_change_idx,
        )
        return skipped_relation_flag, skip_chain_flag, outcome_consistent

    # verifies the observed action/outcome pair can be produced by this chain
    @staticmethod
    def check_outcome_consistency(
        causal_chain_space, chain_chain, causal_observation, chain_idx
    ):
        # check the change induced by this chain vs. observed change
        chain_action = chain_chain[chain_idx].action
        chain_attributes = chain_chain[chain_idx].attributes
        chain_causal_relation_type = chain_chain[chain_idx].causal_relation_type
        chain_precondition = chain_chain[chain_idx].precondition
        # if the actions match and the outcomes match, this chain is consistent with experience
        # consistent with experience means the chain encodes the same state and the same outcome/transition under the same action
        if (
            chain_precondition == causal_observation.causal_relation.precondition
            and chain_action == causal_observation.causal_relation.action
            and chain_causal_relation_type
            == causal_observation.causal_relation.causal_relation_type
            and causal_chain_space.equals(
                chain_attributes, causal_observation.causal_relation.attributes
            )
        ):
            return True
        return False


def prune_perceptual_relations(
    causal_chain_space, trial_selected, perceptual_causal_relations=None
):
    if perceptual_causal_relations is None:
        perceptual_causal_relations = load_perceptually_causal_relations()
    causal_chain_space.prune_space_from_constraints(
        perceptual_causal_relations[trial_selected]
    )
    causal_chain_space.set_uniform_belief_for_causal_chains_with_positive_belief()
    causal_chain_space.write_chains_with_positive_belief(
        "/chains_pruned", batch_size=1000000
    )
    return causal_chain_space


def prune_random_subsample(causal_chain_space, subset_size=10000):
    causal_chain_space.prune_space_random_subset(subset_size=subset_size)
    causal_chain_space.set_uniform_belief_for_causal_chains_with_positive_belief()
    causal_chain_space.write_chains_with_positive_belief("/chains_subsampled")
    return causal_chain_space
