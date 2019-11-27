import time
import math
import copy
import texttable
import random
import numpy as np

# import agent (OpenLockAgents must be in PYTHONPATH)
from openlockagents.agent import Agent

from openlockagents.OpenLockLearner.causal_classes.OutcomeSpace import (
    Outcome,
    OutcomeSpace,
)
from openlockagents.OpenLockLearner.causal_classes.BeliefSpace import (
    AtomicSchemaBeliefSpace,
    AbstractSchemaBeliefSpace,
    InstantiatedSchemaBeliefSpace,
    BottomUpChainBeliefSpace,
    TopDownChainBeliefSpace,
)
from openlockagents.OpenLockLearner.causal_classes.StructureAndBeliefSpaceWrapper import (
    AtomicSchemaStructureAndBeliefWrapper,
    AbstractSchemaStructureAndBeliefWrapper,
    InstantiatedSchemaStructureAndBeliefWrapper,
    TopDownBottomUpStructureAndBeliefSpaceWrapper,
)
from openlockagents.OpenLockLearner.causal_classes.SchemaStructureSpace import (
    AtomicSchemaStructureSpace,
    AbstractSchemaStructureSpace,
    InstantiatedSchemaStructureSpace,
)
from openlockagents.common import DEBUGGING
from openlockagents.OpenLockLearner.util.common import print_message, AblationParams
from openlockagents.OpenLockLearner.learner.CausalLearner import CausalLearner
from openlockagents.OpenLockLearner.learner.InterventionSelector import (
    InterventionSelector,
)
from openlockagents.OpenLockLearner.learner.ModelBasedRL import ModelBasedRLAgent
from openlockagents.OpenLockLearner.util.util import (
    generate_solutions_by_trial_causal_relation,
)
from openlockagents.OpenLockLearner.util.common import (
    verify_valid_probability_distribution,
)

from openlock.common import ENTITY_STATES, Action


class OpenLockLearnerAgent(Agent):
    def __init__(self, env, causal_chain_structure_space, params, **kwargs):
        super(OpenLockLearnerAgent, self).__init__("OpenLockLearner", params, env)
        super(OpenLockLearnerAgent, self).setup_subject(
            human=False, project_src=params["src_dir"]
        )
        self.trial_order = []
        # dicts to keep track of what happened each trial
        self.rewards = dict()
        self.num_chains_with_belief_above_threshold_per_attempt = dict()
        self.attempt_count_per_trial = dict()
        self.num_attempts_between_solutions = dict()
        self.information_gains_per_attempt = dict()
        self.belief_thresholds_per_attempt = dict()
        self.intervention_chain_idxs_per_attempt = dict()
        self.interventions_per_attempt = dict()

        if "print_messages" in params.keys():
            self.print_messages = params["print_messages"]
        else:
            self.print_messages = True
        causal_chain_structure_space.print_messages = self.print_messages

        self.multiproc = params["multiproc"]
        self.deterministic = params["deterministic"]
        self.chain_sample_size = params["chain_sample_size"]
        self.lambda_multiplier = params["lambda_multiplier"]
        self.local_alpha_update = params["local_alpha_update"]
        self.global_alpha_update = params["global_alpha_update"]
        self.ablation = params["ablation_params"]
        self.epsilon = params["epsilon"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_active = params["epsilon_active"]
        self.epsilon_ratios = (
            params["epsilon_ratios"] if "epsilon_ratios" in params else None
        )

        self.causal_learner = CausalLearner(self.print_messages)
        self.intervention_selector = InterventionSelector(
            params["intervention_sample_size"],
            self.print_messages
        )
        self.model_based_agent = None

        # schema managers (structural)
        three_solution_abstract_schema_structure_space = kwargs[
            "three_solution_schemas"
        ]
        two_solution_abstract_schema_structure_space = kwargs["two_solution_schemas"]
        # shuffle the order of the abstract schemas (this help evenly distribute instantiated schemas among cpu cores
        random.shuffle(three_solution_abstract_schema_structure_space.schemas)
        random.shuffle(two_solution_abstract_schema_structure_space.schemas)

        atomic_schema_structure_space = AtomicSchemaStructureSpace()
        instantiated_schema_structure_space = InstantiatedSchemaStructureSpace()

        # belief managers
        bottom_up_components = copy.copy(env.attribute_labels)
        # todo: hacky way of adding action
        bottom_up_components["action"] = ["push", "pull"]
        bottom_up_chain_belief_space = BottomUpChainBeliefSpace(
            len(causal_chain_structure_space),
            bottom_up_components,
            use_indexed_distributions=not self.ablation.INDEXED_DISTRIBUTIONS,
            use_action_distribution=not self.ablation.ACTION_DISTRIBUTION,
        )
        top_down_chain_belief_space = TopDownChainBeliefSpace(
            len(causal_chain_structure_space), init_to_zero=False
        )
        three_solution_abstract_schema_belief_space = AbstractSchemaBeliefSpace(
            len(three_solution_abstract_schema_structure_space)
        )
        two_solution_abstract_schema_belief_space = AbstractSchemaBeliefSpace(
            len(two_solution_abstract_schema_structure_space)
        )
        atomic_schema_belief_space = AtomicSchemaBeliefSpace(
            len(atomic_schema_structure_space)
        )
        instantiated_schema_belief_space = InstantiatedSchemaBeliefSpace(0)

        # pair structures and beliefs
        self.causal_chain_space = TopDownBottomUpStructureAndBeliefSpaceWrapper(
            causal_chain_structure_space,
            bottom_up_chain_belief_space,
            top_down_chain_belief_space,
        )
        self.three_solution_abstract_schema_space = AbstractSchemaStructureAndBeliefWrapper(
            three_solution_abstract_schema_structure_space,
            three_solution_abstract_schema_belief_space,
        )
        self.two_solution_abstract_schema_space = AbstractSchemaStructureAndBeliefWrapper(
            two_solution_abstract_schema_structure_space,
            two_solution_abstract_schema_belief_space,
        )
        self.instantiated_schema_space = InstantiatedSchemaStructureAndBeliefWrapper(
            instantiated_schema_structure_space, instantiated_schema_belief_space
        )
        self.atomic_schema_space = AtomicSchemaStructureAndBeliefWrapper(
            atomic_schema_structure_space, atomic_schema_belief_space
        )
        self.current_abstract_schema_space = None

        # self.qlearner = qlearner
        # set the belief threshold to be slightly less than uniform belief

        self.outcome_space = [
            self.causal_chain_space.structure_space.get_outcome(x)
            for x in range(len(self.causal_chain_space.structure_space))
        ]

        self.observed_causal_observations = []

        self.attempt_count = 1
        self.trial_count = 1

    def setup_outcome_space(self, states, convert_to_ids=True):
        # num_actions_in_chain = len(
        #     [
        #         x
        #         for x in self.causal_chain_space.base_schema.node_id_to_node_dict.keys()
        #         if re.match(ACTION_REGEX_STR, x)
        #     ]
        # )
        num_states_in_chain = (
            self.causal_chain_space.structure_space.num_subchains_in_chain
        )

        if convert_to_ids:
            states = self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                states, target_type="int"
            )
        # outcome_space = OutcomeSpace(STATES_LABEL, num_states_in_chain)
        # convert outcome and intervention spaces to IDs if causal chain space is defined on IDs
        return OutcomeSpace(states, num_states_in_chain, using_ids=convert_to_ids)

    def setup_trial(
        self, scenario_name, action_limit, attempt_limit, specified_trial=None, **kwargs
    ):
        trial_selected = super(OpenLockLearnerAgent, self).setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial, self.multiproc
        )
        # comment to allow agent to execute the same trial multiple times
        assert (
            trial_selected not in self.trial_order
        ), "Trial selected has already been explored"
        self.rewards[trial_selected] = []
        self.num_chains_with_belief_above_threshold_per_attempt[trial_selected] = []
        self.attempt_count_per_trial[trial_selected] = 0
        self.num_attempts_between_solutions[trial_selected] = []
        self.information_gains_per_attempt[trial_selected] = []
        self.belief_thresholds_per_attempt[trial_selected] = []
        self.intervention_chain_idxs_per_attempt[trial_selected] = []
        self.interventions_per_attempt[trial_selected] = []

        # reset chain beliefs to be uniform, regardless of what occurred last trial
        self.causal_chain_space.bottom_up_belief_space.set_uniform_belief()
        self.causal_chain_space.top_down_belief_space.set_uniform_belief()

        # initialize abstract schema space
        n_solutions = self.env.get_num_solutions()
        self.initialize_abstract_schema_space(self.atomic_schema_space, n_solutions)

        # shuffle causal chains to enforce randomness
        if not self.deterministic:
            self.causal_chain_space.structure_space.shuffle()

        true_chains = generate_solutions_by_trial_causal_relation(
            scenario_name, trial_selected
        )
        self.causal_chain_space.structure_space.set_true_causal_chains(
            true_chains, self.causal_chain_space.bottom_up_belief_space
        )

        # define goal and setup model-based agent
        goal = [("door", ENTITY_STATES["DOOR_OPENED"])]
        self.model_based_agent = ModelBasedRLAgent(
            len(true_chains), goal, lambda_multiplier=self.lambda_multiplier
        )

        self.causal_chain_space.bottom_up_belief_space.attribute_space.initialize_local_attributes(
            trial_selected, use_scaled_prior=True
        )

        self.env.reset()
        # prune chains based on initial observation
        # todo: refactor where this is located/where this happens. ugly to pass around this set
        chain_idxs_pruned_from_initial_observation = self.causal_learner.chain_pruner.prune_chains_from_initial_observation(
            self.causal_chain_space,
            self.env,
            self.trial_count,
            self.attempt_count,
            multiproc=self.multiproc,
            using_ids=self.causal_chain_space.structure_space.using_ids,
        )
        # set uniform belief for chains that survived initial pruning
        self.causal_chain_space.bottom_up_belief_space.set_uniform_belief_for_ele_with_belief_above_threshold(
            self.causal_chain_space.bottom_up_belief_space.belief_threshold,
            multiproc=self.multiproc,
        )
        assert (
            self.verify_true_causal_idxs_have_belief_above_threshold()
        ), "True graphs do not have belief above threshold"

        self.num_chains_with_belief_above_threshold_per_attempt[trial_selected].append(
            self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )

        # todo: initial chain beliefs based on attribute beliefs
        # initialize beliefs based on the attribute beliefs, p(G|T)
        self.causal_chain_space.update_bottom_up_beliefs(
            self.env.attribute_order, trial_selected, multiproc=self.multiproc
        )

        # self.initialize_local_q(trial_selected)
        # self.initialize_local_q2(trial_selected)
        # self.pretty_print_top_k_actions_policy(trial_selected, 10, use_global_Q=True)
        # self.pretty_print_top_k_actions_policy(trial_selected, 10)

        return trial_selected, chain_idxs_pruned_from_initial_observation

    def run_trial_openlock_learner(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        interventions_predefined=None,
        use_random_intervention=False,
        chain_idxs_pruned_from_initial_observation=None,
        intervention_mode=None,
    ):
        if intervention_mode == "attempt":
            self.run_trial_openlock_learner_attempt_intervention(
                trial_selected,
                max_steps_with_no_pruning,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
            )
        elif intervention_mode == "action":
            self.run_trial_openlock_learner_action_intervention(
                trial_selected,
                max_steps_with_no_pruning,
                interventions_predefined=interventions_predefined,
                chain_idxs_pruned_from_initial_observation=chain_idxs_pruned_from_initial_observation,
            )
        else:
            raise ValueError(
                "Unexpected intervention mode. Valid modes are 'action' or 'attempt'"
            )

    def run_trial_openlock_learner_action_intervention(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        interventions_predefined=None,
        use_random_intervention=False,
        chain_idxs_pruned_from_initial_observation=None,
    ):
        self.trial_order.append(trial_selected)

        num_steps_since_last_pruning = 0
        trial_finished = False
        completed_solution_idxs = []
        chain_idxs_pruned_this_trial = (
            chain_idxs_pruned_from_initial_observation
            if chain_idxs_pruned_from_initial_observation is not None
            else set()
        )
        intervention_idxs_executed_this_trial = set()

        # if interventions_predefined is not None:
        #     for intervention in interventions_predefined:
        #         self.execute_attempt_intervention(
        #             intervention=intervention,
        #             trial_selected=trial_selected,
        #             causal_change_index=0,
        #         )

        while not trial_finished:
            start_time = time.time()
            self.env.reset()

            causal_change_idx = 0
            attempt_reward = 0
            intervention_info_gain = 0
            num_chains_pruned_this_attempt = 0

            causal_observations = []
            action_sequence = []
            causal_action_sequence = []
            action_sequences_that_should_be_pruned = set()
            causal_change_idxs = []
            action_beliefs_this_attempt = []

            while not self.env.determine_attempt_finished():
                prev_causal_change_idx = causal_change_idx
                chain_idxs_with_positive_belief, bottom_up_chain_idxs_with_positive_belief, top_down_chain_idxs_with_positive_belief = (
                    self.get_causal_chain_idxs_with_positive_belief()
                )
                # print("CHAINS WITH POSITIVE BELIEF:")
                # self.causal_chain_space.pretty_print_causal_chains(chain_idxs_with_positive_belief)

                e = np.random.sample()
                # if we provide a list of epsilons from human data, put it here
                if self.epsilon_ratios is not None:
                    epsilon = self.epsilon_ratios[self.trial_count - 1]
                else:
                    epsilon = self.epsilon
                # random policy
                if self.epsilon_active and e < epsilon:
                    action = self.model_based_agent.random_action_policy()
                # greedy policy
                else:
                    action, action_beliefs = self.model_based_agent.greedy_action_policy(
                        causal_chain_space=self.causal_chain_space,
                        causal_chain_idxs=chain_idxs_with_positive_belief,
                        causal_change_idx=causal_change_idx,
                        action_sequence=causal_action_sequence,
                        intervention_idxs_executed=self.intervention_chain_idxs_per_attempt[
                            self.current_trial_name
                        ],
                        interventions_executed=self.interventions_per_attempt[
                            self.current_trial_name
                        ],
                        first_agent_trial=self.determine_first_trial(),
                        ablation=self.ablation,
                    )
                    # if action_beliefs is an empty dict, we picked a random action
                    action_beliefs_this_attempt.append(action_beliefs)

                action_sequence.append(action)
                causal_change_idxs.append(causal_change_idx)

                # if DEBUGGING:
                #     print_message(
                #         self.trial_count,
                #         self.attempt_count,
                #         "Optimal intervention is {} with info gain {:0.4f}. Took {:0.6f} seconds".format(
                #             action, intervention_info_gain, time.time() - start_time
                #         ),
                #     )

                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Intervention selection took {:0.6f}s and selected intervention: {}".format(
                        time.time() - start_time, action
                    ),
                    self.print_messages
                )

                # execute action
                reward, state_prev, state_cur = self.execute_action_intervention(action)
                causal_change_idx = self.update_bottom_up_attribute_beliefs(
                    action, trial_selected, causal_change_idx
                )

                attempt_reward += reward

                # if we experienced a causal change, append the action to the causal action sequence
                if prev_causal_change_idx != causal_change_idx:
                    causal_action_sequence.append(action)
                else:
                    action_sequences_that_should_be_pruned.add(
                        tuple(causal_action_sequence + [action])
                    )

                # create observations from outcome
                causal_observations = self.causal_learner.create_causal_observation(
                    self.env,
                    action,
                    state_cur,
                    state_prev,
                    causal_observations,
                    causal_change_idx,
                    self.trial_count,
                    self.attempt_count,
                )

                self.causal_chain_space.structure_space.pretty_print_causal_observations(
                    [causal_observations[-1]],
                    print_messages=self.print_messages
                )

                # update causal models
                map_chains, num_chains_pruned_this_action, chain_idxs_consistent, chain_idxs_pruned = self.causal_learner.update_bottom_up_causal_model(
                    env=self.env,
                    causal_chain_space=self.causal_chain_space,
                    causal_chain_idxs=bottom_up_chain_idxs_with_positive_belief,
                    # only take the last observation - we've already checked the previous ones
                    causal_observations=[causal_observations[-1]],
                    action_sequences_to_prune=action_sequences_that_should_be_pruned,
                    trial_name=trial_selected,
                    trial_count=self.trial_count,
                    attempt_count=self.attempt_count,
                    prune_inconsitent_chains=not self.ablation.PRUNING,
                    multiproc=self.multiproc,
                )
                num_chains_pruned_this_attempt += num_chains_pruned_this_action

                chain_idxs_pruned_this_trial.update(chain_idxs_pruned)
                assert (
                    self.verify_true_causal_idxs_have_belief_above_threshold()
                ), "True causal chain idx had belief drop below 0!"

                # update the top-down model based on which chains were pruned
                if len(self.instantiated_schema_space.structure_space) > 0:
                    self.instantiated_schema_space.update_instantiated_schema_beliefs(
                        chain_idxs_pruned, multiproc=self.multiproc
                    )
                    self.causal_chain_space.update_top_down_beliefs(
                        self.instantiated_schema_space, multiproc=self.multiproc
                    )
                    assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
                        self.causal_chain_space.structure_space.true_chain_idxs
                    ), "True chains not in instantiated schemas!"

            # if we are debugging, using pruning, and have action sequences that should be pruned, verify they are
            if DEBUGGING and action_sequences_that_should_be_pruned and not self.ablation.PRUNING:
                # verify that all chain idxs that should be pruned are actually pruned
                assert all(
                    [
                        self.causal_chain_space.bottom_up_belief_space[pruned_idx] == 0
                        for pruned_seq in action_sequences_that_should_be_pruned
                        for pruned_idx in self.causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                            pruned_seq
                        )
                    ]
                ), "action sequence that should be pruned is not!"

            # convert to tuple for hashability
            action_sequence = tuple(action_sequence)

            # determine casual chain executed
            intervention_chain_idxs = self.causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                action_sequence
            )
            # check to make sure the intervention indices executed this attempt are not in the set we have already executed
            assert not intervention_chain_idxs.intersection(
                intervention_idxs_executed_this_trial
            ), "Executing same intervention twice"
            intervention_chain_idxs = list(intervention_chain_idxs)

            # decay epsilon
            if self.epsilon_active:
                self.epsilon = self.epsilon * self.epsilon_decay

            intervention_idxs_executed_this_trial.update(intervention_chain_idxs)

            self.intervention_chain_idxs_per_attempt[self.current_trial_name].append(
                intervention_chain_idxs
            )
            self.interventions_per_attempt[self.current_trial_name].append(
                action_sequence
            )
            self.information_gains_per_attempt[self.current_trial_name].append(
                float(intervention_info_gain)
            )

            prev_num_solutions_remaining = self.env.get_num_solutions_remaining()

            # finish the attempt in the environment
            self.finish_attempt()

            if self.print_messages:
                print("CAUSAL OBSERVATION:")
                self.causal_chain_space.structure_space.pretty_print_causal_observations(
                    causal_observations,
                    print_messages=self.print_messages
                )

            num_solutions_remaining = self.env.get_num_solutions_remaining()
            # solution found, instantiate schemas
            if prev_num_solutions_remaining != num_solutions_remaining:
                self.instantiate_schemas(
                    num_solutions_in_trial=self.env.get_num_solutions(),
                    causal_observations=causal_observations,
                    solution_action_sequence=action_sequence,
                    completed_solution_idxs=completed_solution_idxs,
                    excluded_chain_idxs=chain_idxs_pruned_this_trial,
                    num_solutions_remaining=num_solutions_remaining,
                    multiproc=self.multiproc,
                )

            # self.pretty_print_last_results()

            self.print_attempt_update(
                action_sequence,
                attempt_reward,
                num_chains_pruned_this_attempt,
                model_based_solution_chain_idxs=[],
                map_chains=map_chains,
            )

            # chain_verified, correct_chain_subchain_count = self.causal_chain_space.verify_post_intervention_all_chains_pruned_but_correct_causal_chain(
            #     intervention_id
            # )
            # chains_with_intervention = self.causal_chain_space.get_all_chains_with_actions(
            #     intervention_id
            # )
            # if chain_executed is None:
            #     for chain_with_intervention in chains_with_intervention:
            #         self.causal_chain_space.pretty_print_causal_chains([chain_with_intervention])
            #         assert chain_with_intervention.belief == 0

            self.finish_attempt_openlock_agent(
                trial_selected,
                prev_num_solutions_remaining,
                # chain_executed,
                attempt_reward,
            )

            # todo: refactor to find better way to get trial success (and when this value will be available/set in the trial)
            trial_finished = self.env.get_trial_success()

            # self.verify_true_chains_have_positive_belief()

            # num_chains_pruned.append(num_chains_pruned_this_step)
            # fig = plot_num_pruned(num_chains_pruned, self.writer.subject_path + '/pruning_plot.png')

            if num_chains_pruned_this_attempt == 0:
                num_steps_since_last_pruning += 1
            else:
                num_steps_since_last_pruning = 0

            if (
                num_steps_since_last_pruning > max_steps_with_no_pruning
                and not self.ablation.PRUNING
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Exiting trial due to no chains pruned in {} steps".format(
                        num_steps_since_last_pruning
                    ),
                    self.print_messages
                )
                trial_finished = True
            # else:
            # prevent trial from finishing; continue exploration even if we found all solutions
            # trial_finished = False

        self.finish_trial(trial_selected, test_trial=False)

    def run_trial_openlock_learner_attempt_intervention(
        self,
        trial_selected,
        max_steps_with_no_pruning,
        interventions_predefined=None,
        use_random_intervention=False,
        chain_idxs_pruned_from_initial_observation=None,
    ):
        self.trial_order.append(trial_selected)

        num_steps_since_last_pruning = 0
        trial_finished = False
        completed_solution_idxs = []
        chain_idxs_pruned_across_all_attempts = (
            chain_idxs_pruned_from_initial_observation
            if chain_idxs_pruned_from_initial_observation is not None
            else set()
        )
        intervention_idxs_executed_this_trial = set()
        # loop for attempts in this trial
        while not trial_finished:
            start_time = time.time()
            self.env.reset()
            causal_change_idx = 0

            causal_chain_idxs_with_positive_belief = (
                self.get_causal_chain_idxs_with_positive_belief()
            )
            # print("CHAINS WITH POSITIVE BELIEF:")
            # self.causal_chain_space.pretty_print_causal_chains(causal_chain_idxs_with_positive_belief)

            e = np.random.sample()
            # if we provide a list of epsilons from human data, put it here
            if self.epsilon_ratios is not None:
                epsilon = self.epsilon_ratios[self.trial_count - 1]
            else:
                epsilon = self.epsilon
            # random policy
            if self.epsilon_active and e < epsilon:
                intervention_chain_idx = self.model_based_agent.random_chain_policy(
                    causal_chain_idxs_with_positive_belief
                )
            # greedy policy
            else:
                intervention_chain_idx = self.model_based_agent.greedy_chain_policy(
                    causal_chain_space=self.causal_chain_space,
                    causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                    intervention_idxs_executed=self.intervention_chain_idxs_per_attempt[
                        self.current_trial_name
                    ],
                    interventions_executed=self.interventions_per_attempt[
                        self.current_trial_name
                    ],
                    first_agent_trial=self.determine_first_trial(),
                    ablation=self.ablation,
                )
                assert (
                    intervention_chain_idx not in intervention_idxs_executed_this_trial
                ), "Intervention index already selected, should never select the same index twice"

            # decay epsilon
            if self.epsilon_active:
                self.epsilon = self.epsilon * self.epsilon_decay

            # force-execute a solution
            # if self.attempt_count == 1:
            #     intervention_chain_idx = self.causal_chain_space.true_chain_idxs[0]

            intervention_idxs_executed_this_trial.add(intervention_chain_idx)
            intervention = self.causal_chain_space.structure_space.get_actions(
                intervention_chain_idx
            )
            intervention_info_gain = 0

            # terminating codition; we have exhaustively explored remaining causal chain space
            if intervention is None:
                self.print_complete_exploration_message(
                    causal_chain_idxs_with_positive_belief
                )
                break

            self.intervention_chain_idxs_per_attempt[self.current_trial_name].append(
                intervention_chain_idx
            )
            self.interventions_per_attempt[self.current_trial_name].append(intervention)
            self.information_gains_per_attempt[self.current_trial_name].append(
                float(intervention_info_gain)
            )

            if DEBUGGING:
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Optimal intervention is {} with info gain {:0.4f}. Took {:0.6f} seconds".format(
                        intervention, intervention_info_gain, time.time() - start_time
                    ),
                    self.print_messages
                )

            print_message(
                self.trial_count,
                self.attempt_count,
                "Intervention selection took {:0.6f}s and selected intervention: {}".format(
                    time.time() - start_time, intervention
                ),
                self.print_messages
            )

            prev_num_solutions_remaining = self.env.get_num_solutions_remaining()

            intervention_outcomes, intervention_reward, causal_change_idx = self.execute_attempt_intervention(
                intervention, trial_selected, causal_change_idx
            )
            causal_observations = self.causal_learner.create_causal_observations(
                env=self.env,
                action_sequence=intervention,
                intervention_outcomes=intervention_outcomes,
                trial_count=self.trial_count,
                attempt_count=self.attempt_count,
            )
            attempt_reward = intervention_reward
            # finish the attempt in the environment
            self.finish_attempt()

            if self.print_messages:
                print("CAUSAL OBSERVATION:")
                self.causal_chain_space.structure_space.pretty_print_causal_observations(
                    causal_observations,
                    print_messages=self.print_messages
                )

            num_solutions_remaining = self.env.get_num_solutions_remaining()
            # solution found, instantiate schemas
            if prev_num_solutions_remaining != num_solutions_remaining:
                completed_solution_idxs = self.process_solution(
                    causal_observations=causal_observations,
                    completed_solution_idxs=completed_solution_idxs,
                    solution_action_sequence=intervention,
                )
                self.instantiate_schemas(
                    num_solutions_in_trial=self.env.get_num_solutions(),
                    completed_solutions=completed_solution_idxs,
                    exclude_chain_idxs=chain_idxs_pruned_across_all_attempts,
                    num_solutions_remaining=num_solutions_remaining,
                    multiproc=self.multiproc,
                )

            # self.pretty_print_last_results()

            # update beliefs, uses self's log to get executed interventions/outcomes among ALL chains with with positive belief (even those below threshold)
            map_chains, num_chains_pruned_this_attempt, chain_idxs_consistent, chain_idxs_pruned = self.causal_learner.update_bottom_up_causal_model(
                env=self.env,
                causal_chain_space=self.causal_chain_space,
                causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                causal_observations=causal_observations,
                trial_name=trial_selected,
                trial_count=self.trial_count,
                attempt_count=self.attempt_count,
                prune_inconsitent_chains=not self.ablation.PRUNING,
                multiproc=self.multiproc,
            )
            chain_idxs_pruned_across_all_attempts.update(chain_idxs_pruned)

            assert (
                self.verify_true_causal_idxs_have_belief_above_threshold()
            ), "True causal chain idx had belief drop below 0!"

            # update the top-down model based on which chains were pruned
            if len(self.instantiated_schema_space.structure_space) > 0:
                self.instantiated_schema_space.update_instantiated_schema_beliefs(
                    chain_idxs_pruned, multiproc=self.multiproc
                )
                self.causal_chain_space.update_top_down_beliefs(
                    self.instantiated_schema_space, multiproc=self.multiproc
                )
                assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
                    self.causal_chain_space.structure_space.true_chain_idxs
                ), "True chains not in instantiated schemas!"

            self.print_attempt_update(
                intervention,
                attempt_reward,
                num_chains_pruned_this_attempt,
                model_based_solution_chain_idxs=[],
                map_chains=map_chains,
            )

            # chain_verified, correct_chain_subchain_count = self.causal_chain_space.verify_post_intervention_all_chains_pruned_but_correct_causal_chain(
            #     intervention_id
            # )
            # chains_with_intervention = self.causal_chain_space.get_all_chains_with_actions(
            #     intervention_id
            # )
            # if chain_executed is None:
            #     for chain_with_intervention in chains_with_intervention:
            #         self.causal_chain_space.pretty_print_causal_chains([chain_with_intervention])
            #         assert chain_with_intervention.belief == 0

            self.finish_attempt_openlock_agent(
                trial_selected,
                prev_num_solutions_remaining,
                # chain_executed,
                attempt_reward,
            )

            # todo: refactor to find better way to get trial success (and when this value will be available/set in the trial)
            trial_finished = self.env.get_trial_success()

            # self.verify_true_chains_have_positive_belief()

            # num_chains_pruned.append(num_chains_pruned_this_step)
            # fig = plot_num_pruned(num_chains_pruned, self.writer.subject_path + '/pruning_plot.png')

            if num_chains_pruned_this_attempt == 0:
                num_steps_since_last_pruning += 1
            else:
                num_steps_since_last_pruning = 0

            if (
                num_steps_since_last_pruning > max_steps_with_no_pruning
                and not self.ablation.PRUNING
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "Exiting trial due to no chains pruned in {} steps".format(
                        num_steps_since_last_pruning
                    ),
                    self.print_messages
                )
                trial_finished = True
            # else:
            # prevent trial from finishing; continue exploration even if we found all solutions
            # trial_finished = False

        self.finish_trial(trial_selected, test_trial=False)

    def execute_attempt_intervention(
        self, intervention, trial_selected, causal_change_index
    ):
        intervention_reward = 0
        intervention_outcomes = []
        # execute the intervention in the simulator
        for action in intervention:
            if isinstance(action, str):
                action = Action(action)

            reward, state_prev, state_cur = self.execute_action_intervention(action)
            causal_change_index = self.update_bottom_up_attribute_beliefs(
                action, trial_selected, causal_change_index
            )

            intervention_reward += reward
            intervention_outcomes.append((state_prev, state_cur))

        return intervention_outcomes, intervention_reward, causal_change_index

    def execute_action_intervention(self, action):
        if isinstance(action, str):
            action_env = self.env.action_map[action]
        elif isinstance(action, Action):
            action_env = self.env.action_map[action.name + "_" + action.obj]
        else:
            raise TypeError("Unexpected action type")

        print_message(
            self.trial_count, self.attempt_count, "Executing action: {}".format(action), self.print_messages
        )
        next_state, reward, done, opt = self.env.step(action_env)

        attempt_results = self.get_last_results()

        state_prev = Outcome.parse_results_into_outcome(attempt_results, idx=-2)
        state_cur = Outcome.parse_results_into_outcome(attempt_results, idx=-1)

        # print("Previous state: {}".format(state_prev))
        # print("Current state: {}".format(state_cur))

        return reward, state_prev, state_cur

    def update_bottom_up_attribute_beliefs(
        self, action, trial_selected, causal_change_index
    ):
        # if env changed state, accept this observation into attribute model and update attribute model
        if self.env.determine_fluent_change():
            obj = action.obj
            attributes = self.env.get_obj_attributes(obj)
            if self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids:
                for attribute in attributes:
                    attributes[
                        attribute
                    ] = self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_to_target_type(
                        attribute, attributes[attribute], target_type="int"
                    )
            attributes_to_add = [(name, value) for name, value in attributes.items()]
            # todo: hacky way to add action
            attributes_to_add.append(('action', action.name))
            self.causal_chain_space.bottom_up_belief_space.attribute_space.add_frequencies(
                attributes_to_add,
                trial_selected,
                causal_change_index,
                global_alpha_increase=self.global_alpha_update,
                local_alpha_increase=self.local_alpha_update,
            )
            causal_change_index += 1

        return causal_change_index

    def update_instantiated_schema_beliefs(self, chain_idxs_pruned):
        self.instantiated_schema_space.update_instantiated_schema_beliefs(
            chain_idxs_pruned, multiproc=self.multiproc
        )
        self.causal_chain_space.update_top_down_beliefs(
            self.instantiated_schema_space, multiproc=self.multiproc
        )
        assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
            self.causal_chain_space.structure_space.true_chain_idxs
        ), "True chains not in instantiated schemas!"

    def initialize_abstract_schema_beliefs(self, n_solutions):
        if n_solutions == 2:
            self.two_solution_abstract_schema_space.initialize_abstract_schema_beliefs(
                self.atomic_schema_space
            )
        elif n_solutions == 3:
            self.three_solution_abstract_schema_space.initialize_abstract_schema_beliefs(
                self.atomic_schema_space
            )
        else:
            raise ValueError(
                "Incorrect number of solutions found than accounted for by schemas. Cannot update schema belief"
            )

    def update_atomic_schema_beliefs(self, completed_solutions):
        """
        Update the belief in schemas based on the completed solutions
        :param completed_solutions: list of completed solutions executed
        :return: Nothing
        """
        # truncate pushing on the door from solutions
        truncated_solutions = [
            completed_solution[: len(completed_solution) - 1]
            for completed_solution in completed_solutions
        ]
        self.atomic_schema_space.update_atomic_schema_beliefs(truncated_solutions)
        # deprecated, version without atomic structures at root
        # two solution schema (3 levers)
        # if len(completed_solutions) == 2:
        #     self.two_solution_abstract_schema_space.update_abstract_schema_beliefs(
        #         completed_solutions
        #     )
        # # three solution schema (4 levers)
        # elif len(completed_solutions) == 3:
        #     self.three_solution_abstract_schema_space.update_abstract_schema_beliefs(
        #         completed_solutions
        #     )
        # else:
        #     raise ValueError(
        #         "Incorrect number of solutions found than accounted for by schemas. Cannot update schema belief"
        #     )

    def process_solution(
        self, causal_observations, completed_solution_idxs, solution_action_sequence
    ):
        # construct solution chain from outcome and intervention
        # the intervention we executed is not necessarily the same index as the causal chain observed - multiple causal chains could have the same action sequence
        solution_chain = tuple([x.causal_relation for x in causal_observations])
        true_chain_idx = self.causal_chain_space.structure_space.true_chains.index(
            solution_chain
        )
        solution_causal_chain_idx = self.causal_chain_space.structure_space.true_chain_idxs[
            true_chain_idx
        ]
        completed_solution_idxs.append(solution_causal_chain_idx)
        # update the state of the model-based planner
        self.model_based_agent.update_state(
            self.env.get_num_solutions_remaining(), solution_action_sequence
        )
        return completed_solution_idxs

    def instantiate_schemas(
        self,
        num_solutions_in_trial,
        causal_observations,
        completed_solution_idxs,
        solution_action_sequence,
        excluded_chain_idxs,
        num_solutions_remaining,
        multiproc=True,
    ):
        if num_solutions_remaining <= 0:
            return
        completed_solution_idxs = self.process_solution(
            causal_observations=causal_observations,
            completed_solution_idxs=completed_solution_idxs,
            solution_action_sequence=solution_action_sequence,
        )

        # todo: make every else multiproc compliant - currently there are bugs if other functions are multiprocessed
        instantiated_schemas, instantiated_schema_beliefs = self.current_abstract_schema_space.instantiate_schemas(
            solutions_executed=completed_solution_idxs,
            n_chains_in_schema=num_solutions_in_trial,
            causal_chain_structure_space=self.causal_chain_space.structure_space,
            excluded_chain_idxs=excluded_chain_idxs,
            multiproc=True,
        )

        self.instantiated_schema_space.structure_space = instantiated_schemas
        self.instantiated_schema_space.belief_space.beliefs = (
            instantiated_schema_beliefs
        )

        self.instantiated_schema_space.belief_space.renormalize_beliefs(
            multiproc=self.multiproc
        )

        # update the top-down beliefs
        self.causal_chain_space.update_top_down_beliefs(
            self.instantiated_schema_space, multiproc=multiproc
        )

        # verify true assignment is in schema space
        assert self.instantiated_schema_space.structure_space.verify_chain_assignment_in_schemas(
            self.causal_chain_space.structure_space.true_chain_idxs
        ), "True chains not in instantiated schemas!"

    def initialize_abstract_schema_space(self, atomic_schema_space, n_solutions):
        if n_solutions == 2:
            self.current_abstract_schema_space = self.two_solution_abstract_schema_space
        elif n_solutions == 3:
            self.current_abstract_schema_space = (
                self.three_solution_abstract_schema_space
            )
        else:
            raise ValueError("Incorrect number of solutions")

        self.current_abstract_schema_space.update_abstract_schema_beliefs(
            atomic_schema_space
        )

    def update_trial(self, trial, is_current_trial):
        if is_current_trial:
            self.logger.cur_trial = trial
        else:
            self.logger.cur_trial = None
            self.logger.trial_seq.append(trial)

    def get_causal_chain_idxs_with_positive_belief(self):
        # get indices with positive belief from top-down and bottom-up
        bottom_up_causal_chain_idxs_with_positive_belief = (
            self.causal_chain_space.bottom_up_belief_space.get_idxs_with_belief_above_threshold(print_msg=self.print_messages)
        )
        top_down_causal_chain_idxs_with_positive_belief = (
            self.causal_chain_space.top_down_belief_space.get_idxs_with_belief_above_threshold(print_msg=self.print_messages)
        )
        # actual candidate set is the intersection between top down and bottom up
        causal_chain_idxs_with_positive_belief = list(
            set(bottom_up_causal_chain_idxs_with_positive_belief).intersection(
                set(top_down_causal_chain_idxs_with_positive_belief)
            )
        )

        assert (
            causal_chain_idxs_with_positive_belief
        ), "No causal chains with positive belief in both top-down and bottom-up belief spaces"
        return (
            causal_chain_idxs_with_positive_belief,
            bottom_up_causal_chain_idxs_with_positive_belief,
            top_down_causal_chain_idxs_with_positive_belief,
        )

    def get_true_interventions_and_outcomes(self):
        # Optimal interventions and outcomes
        true_optimal_interventions = [
            solution.actions
            for solution in self.causal_chain_space.structure_space.true_chains
        ]
        true_optimal_outcomes = [
            Outcome(
                solution.states,
                self.intervention_selector.simulate_intervention(
                    solution, solution.actions
                ),
            )
            for solution in self.causal_chain_space.structure_space.true_chains
        ]

        return true_optimal_interventions, true_optimal_outcomes

    def finish_attempt(self):
        """
        finish the attempt in the environment. To be run before the agent updates its internal model
        :return:
        """
        # todo: see if this function should be consolidated with finish_attempt_openlock_agent()
        super(OpenLockLearnerAgent, self).finish_attempt()

    def finish_attempt_openlock_agent(
        self, trial_name, prev_num_solutions_remaining, attempt_reward
    ):
        """
        Finishes the attempt for the openlock agent. To be run after the agent has updated its internal model
        :param trial_name:
        :param prev_num_solutions_remaining:
        :param attempt_reward:
        :return:
        """
        num_solutions_remaining = self.env.get_num_solutions_remaining()
        self.rewards[self.current_trial_name].append(attempt_reward)
        self.belief_thresholds_per_attempt[self.current_trial_name].append(
            self.causal_chain_space.bottom_up_belief_space.belief_threshold
        )
        self.num_chains_with_belief_above_threshold_per_attempt[
            self.current_trial_name
        ].append(
            self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
        )
        # solution found, add to number of attempts between solutions
        if num_solutions_remaining != prev_num_solutions_remaining:
            num_attempts_since_last_solution = (
                self.env.cur_trial.num_attempts_since_last_solution_found
            )
            self.num_attempts_between_solutions[self.current_trial_name].append(
                num_attempts_since_last_solution
            )

        # # chain_idx_executed should only be None if we haven't finished an action sequence (i.e. intervention)
        # if chain_idx_executed is not None:
        #     num_solutions_remaining = self.env.get_num_solutions_remaining()
        #     start_time = time.time()
        #     print_message('Updating Q-learner...')
        #     self.update_qleaner(trial_name, prev_num_solutions_remaining, chain_idx_executed, chain_executed, attempt_reward, num_solutions_remaining)
        #     print_message('Updating Q-learner took {:0.6f}s'.format(time.time() - start_time))
        #     self.pretty_print_top_k_actions_policy(trial_name, 10, use_global_Q=True)
        #     self.pretty_print_top_k_actions_policy(trial_name, 10)
        # else:
        #     print_message('No TD update for Q-learner; no chains remaining with this action sequence')

        self.observed_causal_observations = []
        self.attempt_count += 1

    def finish_trial(self, trial_selected, test_trial):
        super(OpenLockLearnerAgent, self).finish_trial(trial_selected, test_trial)

        self.update_atomic_schema_beliefs(self.env.cur_trial.completed_solutions)

        if DEBUGGING:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_global_attributes()
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_local_attributes(
                self.current_trial_name
            )
            self.current_abstract_schema_space.structure_space.pretty_print(
                self.current_abstract_schema_space.belief_space
            )
            self.atomic_schema_space.structure_space.pretty_print(
                self.atomic_schema_space.belief_space
            )

        # reset instantiated schema space
        self.instantiated_schema_space.structure_space.reset()
        self.instantiated_schema_space.belief_space.reset()

        # minus 1 because we incremented the attempt count
        self.attempt_count_per_trial[self.current_trial_name] = self.attempt_count - 1
        self.attempt_count = 1
        self.trial_count += 1

    def finish_subject(
        self, strategy="OpenLockLearner", transfer_strategy="OpenLockLearner"
    ):
        """
        Prepare agent to save to JSON. Any data to be read in matlab should be converted to native python lists (instead of numpy) before running jsonpickle
        :param strategy:
        :param transfer_strategy:
        :return:
        """
        agent_cpy = copy.copy(self)

        # we need to deep copy the causal chain space so causal_chains is not deleted (and is usable for other agents)
        causal_chain_space_structure = copy.deepcopy(
            agent_cpy.causal_chain_space.structure_space
        )
        agent_cpy.causal_chain_space.structure_space = causal_chain_space_structure

        # todo: this is used to save numpy arrays into json pickle...this should as numpy arrays, but it does not
        agent_cpy.causal_chain_space.bottom_up_belief_space.attribute_space.convert_to_list()
        # cleanup agent for writing to file (deleting causal chain structures and their beliefs)
        # keep bottom_up_belief_space for attributes
        attributes_to_delete = ["structure_space", "top_down_belief_space"]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(agent_cpy.causal_chain_space, attribute_to_delete):
                delattr(agent_cpy.causal_chain_space, attribute_to_delete)
        attributes_to_delete = [
            "outcome_space",
            "cached_outcome_likelihood_sum",
            "cached_outcome_likelihoods_given_intervention_and_chain_times_belief",
        ]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(agent_cpy, attribute_to_delete):
                delattr(agent_cpy, attribute_to_delete)
        # keep bottom_up_belief_space for attribute space
        attributes_to_delete = ["beliefs", "num_idxs_with_belief_above_threshold"]
        for attribute_to_delete in attributes_to_delete:
            if hasattr(
                agent_cpy.causal_chain_space.bottom_up_belief_space, attribute_to_delete
            ):
                delattr(
                    agent_cpy.causal_chain_space.bottom_up_belief_space,
                    attribute_to_delete,
                )

        if hasattr(agent_cpy, "ablation"):
            agent_cpy.ablation = agent_cpy.ablation.__dict__
        if hasattr(agent_cpy, "params") and "ablation" in agent_cpy.params.keys():
            agent_cpy.params["ablation"] = agent_cpy.params["ablation"].__dict__

        super(OpenLockLearnerAgent, self).finish_subject(
            strategy, transfer_strategy, agent_cpy
        )
        # manually mark that we have finished this agent so cleanup() is not called
        # (finished is marked to false on every call to setup_trial
        self.finished = True

    @property
    def current_trial_name(self):
        if self.env is not None and self.env.cur_trial is not None:
            return self.env.cur_trial.name
        else:
            return None

    def convert_outcome_space_to_or_from_ids(self, target_type):
        # convert outcome and intervention spaces to IDs if causal chain space is defined on IDs
        # if causal chain states are defined on ints but outcome state is defined on string, we need to convert IDs
        for i in range(len(self.outcome_space)):
            self.outcome_space[
                i
            ].state_ids = self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                self.outcome_space[i].state_ids, target_type
            )
        if target_type == "int":
            self.outcome_space.using_ids = True
        else:
            self.outcome_space.using_ids = False

    def convert_attribute_space_to_or_from_ids(self, target_type):
        self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
            "color"
        ] = list(
            self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_tuple_to_target_type(
                self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
                    "color"
                ],
                target_type,
            )
        )
        self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
            "position"
        ] = list(
            self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                self.causal_chain_space.bottom_up_belief_space.attribute_space.global_attributes.labels[
                    "position"
                ],
                target_type,
            )
        )
        for (
            key
        ) in (
            self.causal_chain_space.bottom_up_belief_space.attribute_space.local_attributes.keys()
        ):
            local_attributes = self.causal_chain_space.bottom_up_belief_space.attribute_space.local_attributes[
                key
            ]
            local_attributes["color"] = list(
                self.causal_chain_space.structure_space.unique_id_manager.convert_attribute_tuple_to_target_type(
                    local_attributes["color"], target_type
                )
            )
            local_attributes["position"] = list(
                self.causal_chain_space.structure_space.unique_id_manager.convert_states_to_target_type(
                    local_attributes["position"], target_type
                )
            )
        if target_type == "int":
            self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids = (
                True
            )
        else:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.using_ids = (
                False
            )

    def verify_true_causal_idxs_have_belief_above_threshold(self):
        return all(
            [
                True
                if self.causal_chain_space.bottom_up_belief_space[x]
                > self.causal_chain_space.bottom_up_belief_space.belief_threshold
                else False
                for x in self.causal_chain_space.structure_space.true_chain_idxs
            ]
        )

    def attempt_sanity_checks(
        self,
        action_sequences_that_should_be_pruned=None,
        intervention_chain_idxs=None,
        intervention_idxs_executed_this_trial=None,
        chain_idxs_pruned_this_trial=None,
    ):
        assert (
            self.verify_true_causal_idxs_have_belief_above_threshold()
        ), "True causal chain idx had belief drop below 0!"

        if DEBUGGING and chain_idxs_pruned_this_trial:
            assert all(
                [
                    self.causal_chain_space.bottom_up_belief_space[pruned_idx] == 0
                    for pruned_idx in chain_idxs_pruned_this_trial
                ]
            ), "Should have pruned chain that has positive belief"

        # verify all action sequences that should be pruned have 0 belief
        if DEBUGGING and action_sequences_that_should_be_pruned:
            # verify that all chain idxs that should be pruned are actually pruned
            assert all(
                [
                    self.causal_chain_space.bottom_up_belief_space[pruned_idx] == 0
                    for pruned_seq in action_sequences_that_should_be_pruned
                    for pruned_idx in self.causal_chain_space.structure_space.find_causal_chain_idxs_with_actions(
                        pruned_seq
                    )
                ]
            ), "action sequence that should be pruned is not!"

        # verify we don't execute the same intervention twice
        if intervention_chain_idxs and intervention_idxs_executed_this_trial:
            try:
                assert not intervention_chain_idxs.intersection(
                    intervention_idxs_executed_this_trial
                ), "Executing same intervention twice"
            except AssertionError:
                print("problem")
                raise AssertionError("Executing the same intervention twice")

    def determine_first_trial(self):
        return True if self.trial_count == 1 else False

    def pretty_print_policy(self, trial_name, use_global_Q=False):
        if use_global_Q:
            policy_type = "GLOBAL"
        else:
            policy_type = "LOCAL"
        chain_idxs, chain_q_values = self.get_greedy_policy(trial_name, use_global_Q)
        print_message(
            self.trial_count,
            self.attempt_count,
            "GREEDY {} POLICY TO FIND BOTH SOLUTIONS:".format(policy_type),
            self.print_messages
        )
        self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            chain_idxs, q_values=chain_q_values, print_messages=self.print_messages
        )

    def print_attempt_update(
        self,
        intervention_str,
        attempt_reward,
        num_chains_pruned_this_attempt,
        model_based_solution_chain_idxs,
        map_chains,
    ):
        if DEBUGGING:
            print(self.env.cur_trial.attempt_seq[-1].action_seq)

        self.plot_reward(attempt_reward, self.total_attempt_count)
        print_message(
            self.trial_count,
            self.attempt_count,
            "{} chains pruned this attempt".format(num_chains_pruned_this_attempt),
            self.print_messages
        )

        if 0 < len(model_based_solution_chain_idxs) < 20:
            if DEBUGGING:
                print_message(
                    self.trial_count, self.attempt_count, "MODEL-BASED SOLUTION CHAINS", self.print_messages
                )
                self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
                    model_based_solution_chain_idxs,
                    self.causal_chain_space.bottom_up_belief_space,
                    print_messages=self.print_messages
                )

        if 0 < len(map_chains) < 20:
            if (
                self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold
                < 20
            ):
                print_message(
                    self.trial_count,
                    self.attempt_count,
                    "CHAINS WITH BELIEF ABOVE {}: {}".format(
                        self.causal_chain_space.bottom_up_belief_space.belief_threshold,
                        self.causal_chain_space.bottom_up_belief_space.num_idxs_with_belief_above_threshold,
                    ),
                    self.print_messages
                )
                self.causal_chain_space.structure_space.print_chains_above_threshold(
                    self.causal_chain_space.bottom_up_belief_space,
                    self.causal_chain_space.bottom_up_belief_space.belief_threshold,
                )
            if DEBUGGING:
                print_message(self.trial_count, self.attempt_count, "MAP CHAINS", self.print_messages)
                self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
                    map_chains, self.causal_chain_space.bottom_up_belief_space, print_messages=self.print_messages
                )

        if attempt_reward > 0:
            print_message(
                self.trial_count,
                self.attempt_count,
                "Executed {} with reward of {}".format(
                    intervention_str, attempt_reward
                ),
                self.print_messages
            )

    def print_num_attempts_per_trial(self):
        table = texttable.Texttable()
        max_num_solutions = max(
            [
                len(self.num_attempts_between_solutions[trial_name])
                for trial_name in self.trial_order
            ]
        )
        chain_content = []
        for trial_name, attempt_count in self.attempt_count_per_trial.items():
            new_chain_content = [trial_name, attempt_count]
            num_attempts_between_solutions = [
                x for x in self.num_attempts_between_solutions[trial_name]
            ]
            if len(num_attempts_between_solutions) != max_num_solutions:
                # add in values for 3-lever vs 4-lever, two additional columns for trial name and total attempt count
                num_attempts_between_solutions.append("N/A")
            new_chain_content.extend(num_attempts_between_solutions)
            chain_content.append(new_chain_content)

        headers = ["trial name", "attempt count"]
        addition_header_content = [
            "solution {}".format(i) for i in range(max_num_solutions)
        ]
        headers.extend(addition_header_content)
        alignment = ["l", "r"]
        alignment.extend(["r" for i in range(max_num_solutions)])
        table.set_cols_align(alignment)
        content = [headers]
        content.extend(chain_content)

        table.add_rows(content)

        widths = [30, 20]
        widths.extend([20 for i in range(max_num_solutions)])

        table.set_cols_width(widths)

        print(table.draw())

    def print_agent_summary(self):
        for trial_name in self.trial_order:
            self.causal_chain_space.bottom_up_belief_space.attribute_space.pretty_print_local_attributes(
                trial_name
            )
        print("TWO SOLUTION SCHEMA SPACE:")
        self.two_solution_abstract_schema_space.structure_space.pretty_print(
            self.two_solution_abstract_schema_space.belief_space
        )
        print("THREE SOLUTION SCHEMA SPACE:")
        self.three_solution_abstract_schema_space.structure_space.pretty_print(
            self.three_solution_abstract_schema_space.belief_space
        )

        print("NUMBER OF ATTEMPTS PER TRIAL:")
        self.print_num_attempts_per_trial()

    def print_complete_exploration_message(
        self, causal_chain_idxs_with_positive_belief
    ):
        print_message(
            self.trial_count,
            self.attempt_count,
            "Causal chain space completely explored. Causally plausible chains:",
            self.print_messages
        )
        self.causal_chain_space.structure_space.pretty_print_causal_chain_idxs(
            causal_chain_idxs_with_positive_belief,
            self.causal_chain_space.bottom_up_belief_space,
        )
        print_message(
            self.trial_count,
            self.attempt_count,
            "Causal chain space completely explored with {} causally plausible chains. Exiting causal learning...".format(
                len(causal_chain_idxs_with_positive_belief)
            ),
            self.print_messages
        )

    def plot_num_pruned(self, num_chains_pruned):
        self.plot_value(
            "Number of chains pruned (log)",
            math.log(num_chains_pruned[-1]) if num_chains_pruned[-1] > 0 else 0,
            len(num_chains_pruned),
        )

    # def pretty_print_top_k_actions_policy(self, trial_name, k, use_global_Q=False):
    #     if not use_global_Q:
    #         chain_idxs, chain_q_values = self.qlearner.get_top_k_actions_local_policy(
    #             trial_name, k
    #         )
    #         policy_type = "LOCAL"
    #     else:
    #         chain_idxs, chain_q_values = self.qlearner.get_top_k_actions_global_policy(
    #             k
    #         )
    #         policy_type = "GLOBAL"
    #     for i in range(len(chain_idxs) - 1, 0, -1):
    #         print_message(
    #             self.trial_count,
    #             self.attempt_count,
    #             "PRINTING {} POLICY TOP {} CHAINS FOR {} WITH {} SOLUTIONS REMAINING".format(
    #                 policy_type, k, trial_name, i
    #             ),
    #         )
    #         self.causal_chain_space.pretty_print_causal_chains(
    #             [self.causal_chain_space.causal_chains[j] for j in chain_idxs[i]],
    #             q_values=chain_q_values[i],
    #         )

    # def update_qleaner(
    #     self,
    #     trial_name,
    #     prev_num_solutions_remaining,
    #     chain_idx_executed,
    #     chain_executed,
    #     attempt_reward,
    #     num_solutions_remaining,
    # ):
    #     td_contribution = self.qlearner.global_td_update(
    #         state=prev_num_solutions_remaining,
    #         action=chain_idx_executed,
    #         reward=attempt_reward,
    #         next_state=num_solutions_remaining,
    #     )
    #     if td_contribution > 0:
    #         # retrospectively distribution reward to similar chains
    #         self.distribute_td_contribution(
    #             self.qlearner.global_Q,
    #             prev_num_solutions_remaining,
    #             chain_executed,
    #             td_contribution,
    #         )
    #     self.qlearner.local_td_update(
    #         trial_name=trial_name,
    #         state=prev_num_solutions_remaining,
    #         action=chain_idx_executed,
    #         reward=attempt_reward,
    #         next_state=num_solutions_remaining,
    #     )
    #
    # def distribute_td_contribution(self, Q, state, spreading_chain, td_contribution):
    #     old_max = Q[state].max()
    #     indexed_confidences, total_confidences = self.compute_indexed_confidences()
    #     for causal_chain in self.causal_chain_space.causal_chains:
    #         similarity = self.compute_chain_similarity(
    #             spreading_chain, causal_chain, indexed_confidences
    #         )
    #         Q[state][causal_chain.chain_id] += similarity * td_contribution
    #     # renormalize s.t. the new max is the same as the old
    #     new_max = Q[state].max()
    #     normalization_factor = new_max / old_max
    #     Q[state] = Q[state] / normalization_factor
    #
    # def get_previous_state(self):
    #     previous_state = self.logger.cur_trial.cur_attempt.results[-1]
    #
    #     return previous_state
    #
    # def get_greedy_policy(self, trial_name, use_global_Q=False):
    #     if not use_global_Q:
    #         chain_idxs, chain_q_values = self.qlearner.get_greedy_local_policy(
    #             trial_name
    #         )
    #     else:
    #         chain_idxs, chain_q_values = self.qlearner.get_greedy_global_policy()
    #     chain_idxs = [
    #         chain_idxs[num_solutions_remaining]
    #         for num_solutions_remaining in list(
    #             reversed(range(1, len(self.env.get_solutions()) + 1))
    #         )
    #     ]
    #     chain_q_values = [
    #         chain_q_values[num_solutions_remaining]
    #         for num_solutions_remaining in list(
    #             reversed(range(1, len(self.env.get_solutions()) + 1))
    #         )
    #     ]
    #     return chain_idxs, chain_q_values
