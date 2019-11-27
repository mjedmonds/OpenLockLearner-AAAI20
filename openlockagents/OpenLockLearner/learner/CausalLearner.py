import time

from openlockagents.OpenLockLearner.util.common import (
    SANITY_CHECK_ELEMENT_LIMIT,
    print_message,
)
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalObservation,
    CausalRelation,
    CausalRelationType,
)
from openlockagents.OpenLockLearner.learner.ChainPruner import ChainPruner
from openlockagents.common import DEBUGGING



class CausalLearner:
    def __init__(self, print_messages=True):
        self.chain_pruner = ChainPruner(print_messages)
        self.print_messages = print_messages

    # updates the learner's model based on the results
    def update_bottom_up_causal_model(
        self,
        env,
        causal_chain_space,
        causal_chain_idxs,
        causal_observations,
        action_sequences_to_prune,
        trial_name,
        trial_count,
        attempt_count,
        prune_inconsitent_chains=True,
        multiproc=False,
    ):
        function_start_time = time.time()

        chain_idxs_consistent = causal_chain_space.bottom_up_belief_space.get_idxs_with_belief_above_threshold(print_msg=False)
        chain_idxs_removed = []

        prev_num_chains_with_belief_above_threshold = (
            causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )

        if prune_inconsitent_chains:
            # todo: this only processes one causal observation; cannot handle multiple fluent changes in one time step
            chain_idxs_consistent, chain_idxs_removed = self.chain_pruner.prune_inconsistent_chains_v2(causal_chain_space=causal_chain_space, causal_chain_idxs=causal_chain_idxs, action_sequences_to_prune=action_sequences_to_prune)
            # chain_idxs_consistent_v1, chain_idxs_removed_v1 = self.chain_pruner.prune_inconsistent_chains(
            #     causal_chain_space,
            #     causal_chain_idxs,
            #     causal_observations,
            #     trial_count,
            #     attempt_count,
            #     multiproc=multiproc,
            # )

        if DEBUGGING:
            print_message(
                trial_count,
                attempt_count,
                "REMOVED {} CHAINS".format(len(chain_idxs_removed)),
                self.print_messages
            )
            # if 0 < len(chain_idxs_removed) < 100:
            #     causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            #         chain_idxs_removed, belief_manager
            #     )

        start_time = time.time()
        # print_message(trial_count, attempt_count, "Updating beliefs...")
        map_chains = causal_chain_space.update_bottom_up_beliefs(
            env.attribute_order, trial_name, multiproc=multiproc
        )
        print_message(
            trial_count,
            attempt_count,
            "Updating beliefs took {:0.6f}s".format(time.time() - start_time),
            self.print_messages
        )

        # the remainder of this function is bookkeeping
        num_chains_with_belief_above_threshold = (
            causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )
        num_chains_pruned = (
            prev_num_chains_with_belief_above_threshold
            - num_chains_with_belief_above_threshold
        )
        if num_chains_pruned < 0:
            start_msg = "Added {} chains above threshold".format(-1 * num_chains_pruned)
        else:
            start_msg = "Eliminated {} chains".format(num_chains_pruned)

        assert num_chains_pruned == len(chain_idxs_removed), "Number of chains removed is incorrect"

        print_message(
            trial_count,
            attempt_count,
            "{}. {} chains pruned across all attempts. Previous number chains above threshold: {} current: {}".format(
                start_msg,
                len(causal_chain_space.structure_space.causal_chains) - num_chains_pruned,
                prev_num_chains_with_belief_above_threshold,
                num_chains_with_belief_above_threshold,
            ),
            self.print_messages
        )
        print_message(
            trial_count,
            attempt_count,
            "Model update/pruning took {:0.6f}s".format(
                time.time() - function_start_time
            ),
            self.print_messages
        )
        return map_chains, num_chains_pruned, chain_idxs_consistent, chain_idxs_removed

    @staticmethod
    def create_causal_observations(
        env, action_sequence, intervention_outcomes, trial_count, attempt_count
    ):
        causal_observations = []
        causal_change_idx = 0

        for i in range(len(action_sequence)):
            action = action_sequence[i]
            prev_state, cur_state = intervention_outcomes[i]
            causal_observations = CausalLearner.create_causal_observation(env, action, cur_state, prev_state, causal_observations, causal_change_idx, trial_count, attempt_count)
        return causal_observations

    @staticmethod
    def create_causal_observation(env, action, cur_state, prev_state, causal_observations, causal_change_idx, trial_count, attempt_count):
        state_diff = cur_state - prev_state
        state_change_occurred = len(state_diff) > 0
        # todo: generalize to more than 1 state change
        if len(state_diff) > 2:
            print_message(
                trial_count,
                attempt_count,
                "More than one state change this iteration, chain assumes only one variable changes at a time: {}".format(
                    state_diff
                ),
            )

        precondition = None
        # need to check for previous effective precondition.
        # We could take an action with an effect, take an action with no effect, then take an action with an effect.
        # We want the precondition to carry over from the first action, so we need to find the preconditon of the last action with an effect
        for i in reversed(range(0, len(causal_observations))):
            if (
                causal_observations[i].causal_relation.causal_relation_type
                is not None
            ):
                precondition = (
                    causal_observations[i].causal_relation.attributes,
                    causal_observations[i].causal_relation.causal_relation_type[1],
                )
                # want the first precondition we find, so break
                break

        if state_change_occurred:
            # todo: refactor to include door_lock
            state_diff = [x for x in state_diff if x[0] != "door_lock"]
            # todo: this only handles a single state_diff per timestep
            assert (
                len(state_diff) < 2
            ), "Multiple fluents changing at each time step not yet implemented"
            state_diff = state_diff[0]
            causal_relation_type = CausalRelationType(state_diff[1])
            attributes = env.get_obj_attributes(state_diff[0])
            attributes = tuple(attributes[key] for key in env.attribute_order)
            causal_observations.append(
                CausalObservation(
                    CausalRelation(
                        action=action,
                        attributes=attributes,
                        causal_relation_type=causal_relation_type,
                        precondition=precondition,
                    ),
                    info_gain=None,
                )
            )
            causal_change_idx += 1

            # self.observed_causal_observations.append(causal_observations[0])
            # if there was a state change increment our state change index/number
            # self.effective_causal_change_idx += 1
        else:
            # prune chains based on action at current state_change_idx_this_attempt
            causal_observations.extend(
                [
                    CausalObservation(
                        CausalRelation(
                            action=action,
                            attributes=None,
                            causal_relation_type=None,
                            precondition=precondition,
                        ),
                        info_gain=None,
                    )
                ]
            )
        return causal_observations
