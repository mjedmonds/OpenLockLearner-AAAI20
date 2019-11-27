import time
import math
import numpy as np

from openlockagents.OpenLockLearner.util.common import print_message


class InterventionSelector:
    def __init__(self, intervention_sample_size, print_messages=True):
        self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief = []
        self.cached_outcome_likelihood_sum = 0.0
        self.intervention_sample_size = intervention_sample_size
        self.print_messages = True

    def select_intervention(
        self,
        causal_chain_space,
        causal_chain_idxs_with_positive_belief,
        outcomes,
        chain_sample_size,
        interventions_per_attempt,
        trial_count,
        attempt_count,
        multiproc,
        sample_chains=True,
        use_random_intervention=False,
    ):
        if use_random_intervention:
            intervention, intervention_info_gain = self.select_intervention_random(
                causal_chain_space=causal_chain_space,
                causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                chain_sample_size=chain_sample_size,
                interventions_executed=interventions_per_attempt,
                trial_count=trial_count,
                attempt_count=attempt_count,
                sample_chains=sample_chains,
            )
        else:
            intervention, intervention_info_gain = self.select_intervention_information_gain(
                causal_chain_space=causal_chain_space,
                causal_chain_idxs=causal_chain_idxs_with_positive_belief,
                outcomes=outcomes,
                chain_sample_size=chain_sample_size,
                interventions_executed=interventions_per_attempt,
                trial_count=trial_count,
                attempt_count=attempt_count,
                sample_chains=sample_chains,
                multiproc=multiproc,
            )
        return intervention, intervention_info_gain

    # chooses a randome intervention
    def select_intervention_random(
        self,
        causal_chain_space,
        causal_chain_idxs,
        chain_sample_size,
        interventions_executed,
        trial_count,
        attempt_count,
        sample_chains=False,
    ):
        selected_causal_chain_idxs = []

        # sanity check
        assert_str = "Causal chain space has negative number ({}) of chains with belief above threshold {}".format(
            causal_chain_space.structure_space.num_chains_with_belief_above_threshold,
            causal_chain_space.structure_space.belief_threshold,
        )
        assert (
            causal_chain_space.structure_space.num_chains_with_belief_above_threshold >= 0
        ), assert_str

        if sample_chains:
            selected_causal_chain_idxs = self.sample_chains(
                causal_chain_space.structure_space,
                causal_chain_idxs,
                chain_sample_size,
                interventions_executed,
            )
        else:
            selected_causal_chain_idxs = causal_chain_idxs
        print_message(
            trial_count,
            attempt_count,
            "Randomly picking intervention using {} chains from {} possible chains.".format(
                len(selected_causal_chain_idxs),
                causal_chain_space.structure_space.num_chains_with_belief_above_threshold,
            ),
            self.print_messages
        )
        rand_idx = np.random.randint(0, len(selected_causal_chain_idxs))
        intervention_idx = selected_causal_chain_idxs[rand_idx]
        intervention = causal_chain_space.structure_space.causal_chains.get_actions(intervention_idx)
        return intervention, 0

    # chooses an intervention
    def select_intervention_information_gain(
        self,
        causal_chain_space,
        causal_chain_idxs,
        outcomes,
        chain_sample_size,
        interventions_executed,
        trial_count,
        attempt_count,
        sample_chains=False,
        multiproc=False,
    ):
        selected_causal_chain_idxs = []

        # sanity check
        assert_str = "Causal chain space has negative number ({}) of chains with belief above threshold {}".format(
            causal_chain_space.bottom_up_belief_space.num_chains_with_belief_above_threshold,
            causal_chain_space.bottom_up_belief_space.belief_threshold,
        )
        assert causal_chain_space.bottom_up_belief_space.num_chains_with_belief_above_threshold >= 0, assert_str

        if sample_chains:
            # sample a set of chains used to perform intervention selection
            selected_causal_chain_idxs, all_chains_executed = self.sample_chains(
                causal_chain_space.structure_space,
                causal_chain_idxs,
                chain_sample_size,
                interventions_executed,
            )
            if all_chains_executed:
                return None, None
        else:
            selected_causal_chain_idxs = causal_chain_idxs

        print_message(
            trial_count,
            attempt_count,
            "Estimating information gain using {} chains from {} possible chains.".format(
                len(selected_causal_chain_idxs),
                causal_chain_space.bottom_up_belief_space.num_chains_with_belief_above_threshold,
            ),
            self.print_messages
        )

        # single processing
        if not multiproc:
            # estimate the information gain from a sample of interventions
            optimal_intervention, optimal_information_gain = self.select_intervention_common(
                causal_chain_space,
                selected_causal_chain_idxs,
                outcomes,
                trial_count,
                attempt_count,
            )
            return optimal_intervention, optimal_information_gain
        # multiprocessing
        else:
            # todo: run single-threaded version since joblib cannot pickle
            optimal_intervention, optimal_information_gain = self.select_intervention_common(
                causal_chain_space,
                selected_causal_chain_idxs,
                outcomes,
                trial_count,
                attempt_count,
            )
            return optimal_intervention, optimal_information_gain
            # todo: fix multiprocessing method below - joblib cannot pickle causal_classes
            # slicing_indices = generate_slicing_indices(interventions)
            # results = mp.ProcessingPool.map(
            #     self.select_intervention_common,
            #     selected_causal_chains,
            #     interventions,
            #     outcomes,
            # )
            # with Parallel(n_jobs=multiprocessing.cpu_count(), verbose=5) as parallel:
            #     intervention_and_information_gain = parallel(
            #         delayed(self.select_intervention_common)(
            #             selected_causal_chains, [interventions[i]], outcomes
            #         )
            #         for i in range(len(interventions))
            #     )
            # optimal_intervention, optimal_information_gain = max(
            #     intervention_and_information_gain, key=itemgetter(1)
            # )
            # return optimal_intervention, optimal_information_gain

    def select_intervention_common(
        self,
        causal_chain_space,
        selected_causal_chain_idxs,
        outcomes,
        trial_count,
        attempt_count,
    ):
        optimal_intervention = None
        optimal_intervention_information_gain = -1  # worst case, do something random
        i_count = 0
        t = time.time()
        while optimal_intervention_information_gain <= 0:
            start_time = time.time()
            print_message(trial_count, attempt_count, "Sampling intervention...", self.print_messages)
            interventions, intervention_chain_idxs, all_chains_executed = self.sample_intervention_batch(
                causal_chain_space.structure_space,
                selected_causal_chain_idxs,
                self.intervention_sample_size,
            )
            assert all(
                [
                    causal_chain_space.bottom_up_belief_space.beliefs[intervention_chain_idx] > 0
                    for intervention_chain_idx in intervention_chain_idxs
                ]
            ), "Intervention does not have positive belief"

            print_message(
                trial_count,
                attempt_count,
                "Sampling intervention took {:0.6f}s".format(time.time() - start_time),
                self.print_messages
            )
            total_interventions = +len(interventions)
            for intervention in interventions:
                intervention_information_gain = 0
                # take expectation of this intervention over possible outcomes
                o_count = 0
                total_num_positive_likelihoods = 0
                for outcome in outcomes:
                    information_gain, num_positive_likelihoods = self.compute_information_gain(
                        causal_chain_space,
                        selected_causal_chain_idxs,
                        intervention,
                        outcome,
                    )
                    total_num_positive_likelihoods += num_positive_likelihoods
                    intervention_information_gain += information_gain
                    o_count += 1
                # todo: why would this be negative?
                if intervention_information_gain < 0:
                    print_message(
                        trial_count,
                        attempt_count,
                        "Intervention information gain is less than 0...",
                    )
                    raise ValueError("Intervention information gain is less than 0")
                if (
                    intervention_information_gain
                    > optimal_intervention_information_gain
                ):
                    optimal_intervention = intervention
                    optimal_intervention_information_gain = (
                        intervention_information_gain
                    )
                # if intervention == tuple_to_check:
                #     print('UL information gain: {}'.format(intervention_information_gain))
                #     information_gain = self.compute_information_gain(selected_causal_chains, intervention, Outcome(state_ids=[7,7,7],state_assignments=[0,1,0]))
                print_message(
                    trial_count,
                    attempt_count,
                    "Computed information gain for all outcomes for intervention {}/{}. {} positive likelihoods computed for intervention {}.".format(
                        i_count + 1,
                        total_interventions,
                        total_num_positive_likelihoods,
                        intervention,
                    ),
                    self.print_messages
                )
                print_message(
                    trial_count,
                    attempt_count,
                    "Gain of {:0.4f} for intervention {}. Optimal intervention is {} with gain of {:0.4f}. Took {:0.6f} seconds".format(
                        intervention_information_gain,
                        intervention,
                        optimal_intervention,
                        optimal_intervention_information_gain,
                        time.time() - t,
                    ),
                    self.print_messages
                )
                t = time.time()
                i_count += 1

            # todo: since we sample interventions only from valid chains, shouldn't we always have a positive information gain?
            # todo: We should be able to learn something by executing a chain from the plausible space, no matter what. Investigate.
            if optimal_intervention_information_gain <= 0:
                # todo: should this be selected_causal_chains or causal_chains_with_positive_belief?
                print_message(
                    trial_count,
                    attempt_count,
                    "No intervention produced information gain. Resampling.",
                    self.print_messages,
                )

        return optimal_intervention, optimal_intervention_information_gain

    @staticmethod
    def sample_chains(
        causal_chain_structure_space, causal_chain_idxs, chain_sample_size, interventions_executed
    ):
        selected_causal_chain_idxs = []
        all_chains_executed = False

        while len(selected_causal_chain_idxs) == 0:
            selected_causal_chain_idxs, all_chains_executed = causal_chain_structure_space.sample_chains(
                causal_chain_idxs,
                sample_size=chain_sample_size,
                action_sequences_executed=interventions_executed,
            )
            if all_chains_executed:
                break

        return selected_causal_chain_idxs, all_chains_executed

    # simulates intervention using this compact causal chain and the schema
    @staticmethod
    def simulate_intervention(causal_chain_structure_space, compact_causal_chain, intervention):
        # verify this chain is capable of performing this intervention (i.e. the action nodes match)
        if compact_causal_chain.actions != intervention:
            return None
        # output the state variable settings according to the CPT settings
        states = []
        for i in range(len(compact_causal_chain.conditional_probability_table_choices)):
            cpt_choice = compact_causal_chain.conditional_probability_table_choices[i]
            state_id = causal_chain_structure_space.conditional_probability_table_combination_labels[
                i
            ]
            # cur fluent value is stored in the rightmost column
            state_outcome = causal_chain_structure_space.base_schema.node_id_to_node_dict[
                state_id
            ].conditional_probability_table[cpt_choice][-1]
            states.append(int(state_outcome))

        return tuple(states)

    # computes p(o|q,g)*p(g), which is fixed in time for every g,q,o pair, also computes the sum of this term
    # due to memory constraints, compute a cache of these values given a specific q and o over all possible chains
    def compute_cached_outcome_likelihoods_given_intervention_and_chain_times_belief(
        self,
        causal_chain_space,
        selected_causal_chain_idxs,
        intervention,
        outcome,
    ):
        num_likelihoods = len(selected_causal_chain_idxs)
        num_positive_likelihoods = 0
        sum_ = 0
        self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief = np.zeros(
            num_likelihoods
        )
        chains_with_positive_likelihoods = []
        # self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief = [
        #     0.0
        # ] * num_likelihoods
        for i in range(len(selected_causal_chain_idxs)):
            causal_chain_idx = selected_causal_chain_idxs[i]
            chain_chain = causal_chain_space.structure_space.causal_chains[causal_chain_idx]
            chain_belief = causal_chain_space.bottom_up_belief_space.beliefs[causal_chain_idx]

            # compute outcome likelihood
            # outcome_likelihood = self.compute_outcome_likelihood_given_intervention_and_chain(causal_chain, intervention, outcome)

            # verify actions and outcomes match
            if all(
                chain_chain[i].action == intervention[i]
                and chain_chain[i].causal_relation_type[1] == outcome[i]
                for i in range(len(chain_chain))
            ):
                outcome_likelihood = 1.0
                num_positive_likelihoods += 1
                chains_with_positive_likelihoods.append(causal_chain_idx)
            else:
                outcome_likelihood = 0.0

            cached_term = outcome_likelihood * chain_belief
            self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief[
                i
            ] = cached_term
            sum_ += cached_term

        self.cached_outcome_likelihood_sum = sum_
        return num_positive_likelihoods

    # computes, p(o|q,g) the likelihood of an outcome given a specific chain and intervention
    # answers whether or not this chain can parse this intervention-outcome pair
    # def compute_outcome_likelihood_given_intervention_and_chain(
    #     self, causal_chain, intervention, outcome
    # ):
    #     # if chain cannot produce this intervention, outcome is impossible
    #     if causal_chain.actions != intervention:
    #         return 0.0
    #     # verify state ids/labels match
    #     if causal_chain.states != outcome.state_ids:
    #         return 0.0
    #     # verify that this chain can produce the state values in the outcome
    #     simulated_outcome = self.simulate_intervention(causal_chain, intervention)
    #     if simulated_outcome != outcome.state_values:
    #         return 0.0
    #     return 1.0

    def compute_information_gain(
        self,
        causal_chain_space,
        selected_causal_chain_idxs,
        intervention,
        outcome,
    ):
        # compute cache of p(o|q,g) for each chain
        num_positive_outcome_likelihoods = self.compute_cached_outcome_likelihoods_given_intervention_and_chain_times_belief(
            causal_chain_space,
            selected_causal_chain_idxs,
            intervention,
            outcome,
        )
        # print('Computed cached p(o|q,g)*p(g) for this intervention. Took {} seconds'.format(time.time() - t))
        outcome_likelihood = 0
        entropy_prior = 0
        entropy_conditional = 0

        for i in range(len(selected_causal_chain_idxs)):
            causal_chain_idx = selected_causal_chain_idxs[i]
            chain_belief = causal_chain_space.bottom_up_belief_space.beliefs[causal_chain_idx]

            if chain_belief <= causal_chain_space.bottom_up_belief_space.belief_threshold:
                continue

            # entropy prior
            # H[G] = \sum_{g\in\Omega G} p(g)*\log p(g)
            entropy_prior += chain_belief + math.log(chain_belief)

            # conditional entropy
            # H[G|q,o] = \sum_{g\in\Omega G} p(g|q,o)*\log p(g|q,o)
            # posterior calculation
            # posterior = self.compute_posterior_given_intervention_and_outcome(i, cached_outcome_likelihoods_given_intervention_and_chain, cached_sum)
            # numerator = p(o|q,g)p(g)
            numerator = self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief[
                i
            ]
            if numerator == 0:
                posterior = 0
            else:
                # denominator = \sum_{g'\in \Omega_G, g' \neq g} p(o|q,g)p(g)
                denominator = self.cached_outcome_likelihood_sum - numerator
                # special case when there is only 1 valid chain
                if denominator == 0:
                    posterior = 1
                else:
                    posterior = numerator / denominator
            if posterior != 0:
                entropy_conditional += posterior * math.log(posterior)

            # p(o|q) = \sum_{g\in\Omega G} p(o|q,g)*p(g)
            # outcome_likelihood += self.compute_outcome_likelihood_given_intervention(cached_sum)
            outcome_likelihood += self.cached_outcome_likelihood_sum

        information_gain = (
            (-1 * entropy_prior) - (-1 * entropy_conditional)
        ) * outcome_likelihood
        # if information_gain < 0:
        #     print('bug')
        return information_gain, num_positive_outcome_likelihoods

    # computes p(g|q,o)
    def compute_posterior_given_intervention_and_outcome(self, causal_chain_idx):
        # numerator = p(o|q,g)p(g)
        numerator = self.cached_outcome_likelihoods_given_intervention_and_chain_times_belief[
            causal_chain_idx
        ]
        if numerator == 0:
            posterior = 0
        else:
            # denominator = \sum_{g'\in \Omega_G} p(o|q,g)p(g)
            denominator = self.cached_outcome_likelihood_sum
            # special case when there is only 1 valid chain
            if denominator == 0:
                posterior = 1.0
            else:
                posterior = numerator / denominator
        return posterior

    # samples interventions from chains that have positive belief
    def sample_intervention_batch(
        self, causal_chain_structure_space, causal_chain_idxs, batch_size=10000
    ):

        if batch_size is None:
            batch_size = causal_chain_structure_space.num_chains_with_belief_above_threshold

        # this encourages a variety of interventions, instead of sticking to MAP interventions when batch_size is close to num_chains_with_positive_belief
        # while (
        #     batch_size >= self.causal_chain_structure_space.num_chains_with_belief_above_threshold / 2
        #     and batch_size > 1
        # ):
        #     # this means there are as many valid indices as chains. We always want to have less sampled intervention_idxs
        #     # than the number of chains in the causal chain space. This means we won't be biased towards a particular chain
        #     # (prevents picking the same intervention every iteration)
        #     batch_size = max(batch_size // 2, 1)

        intervention_chain_idxs, all_chains_executed = causal_chain_structure_space.sample_chains(
            causal_chain_idxs, sample_size=batch_size
        )

        # intervention_chain_ids will be none if we have executed all chains
        if intervention_chain_idxs is not None:
            interventions = [
                causal_chain_structure_space.causal_chains.get_actions(intervention_chain_idx)
                for intervention_chain_idx in intervention_chain_idxs
            ]

        return interventions, intervention_chain_idxs, all_chains_executed
