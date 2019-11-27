import time
import multiprocessing
from joblib import Parallel, delayed
import networkx as nx

import numpy as np

from openlockagents.OpenLockLearner.util.common import (
    SANITY_CHECK_ELEMENT_LIMIT,
    generate_slicing_indices,
    PARALLEL_MAX_NBYTES,
    get_highest_N_idxs,
)

from openlockagents.OpenLockLearner.causal_classes.AttributeSpace import (
    setup_attribute_space,
)
from openlockagents.OpenLockLearner.causal_classes.DirichletDistribution import (
    DirichletDistribution,
)


# helper functions for multiprocessing
def set_num_idxs_with_belief_above_threshold_multiproc(belief_space):
    slicing_indices = generate_slicing_indices(belief_space)
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        summations = parallel(
            delayed(belief_space.compute_num_idxs_with_belief_above_threshold_common)(
                slicing_indices[i - 1], slicing_indices[i]
            )
            for i in range(1, len(slicing_indices))
        )
        summation = sum(summations)
        return summation


def set_uniform_belief_for_idx_with_belief_above_threshold_multiproc(
    belief_space, threshold,
):
    slicing_indices = generate_slicing_indices(belief_space.beliefs)
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        returned_tuples = parallel(
            delayed(
                belief_space.set_uniform_belief_for_idx_with_belief_above_threshold_common
            )(
                slicing_indices[i - 1],
                slicing_indices[i],
                threshold,
            )
            for i in range(1, len(slicing_indices))
        )
        for returned_tuple in returned_tuples:
            returned_beliefs, starting_index, ending_index = returned_tuple
            belief_space.beliefs[starting_index:ending_index] = returned_beliefs
    return belief_space.beliefs


def sum_beliefs_multiproc(belief_space):
    slicing_indices = generate_slicing_indices(belief_space.beliefs)
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        sums = parallel(
            delayed(belief_space.sum_beliefs_common)(
                slicing_indices[i - 1], slicing_indices[i]
            )
            for i in range(1, len(slicing_indices))
        )
        return sum(sums)


def renormalize_beliefs_multiproc(belief_space, normalization_factor):
    belief_space.num_idxs_with_belief_above_threshold = 0
    slicing_indices = generate_slicing_indices(belief_space.beliefs)
    with Parallel(
        n_jobs=multiprocessing.cpu_count(), verbose=5, max_nbytes=PARALLEL_MAX_NBYTES
    ) as parallel:
        max_belief = -1
        max_eles = []
        num_idxs_with_belief_above_threshold = 0
        return_tuples = parallel(
            delayed(belief_space.renormalize_beliefs_common)(
                normalization_factor, slicing_indices[i - 1], slicing_indices[i]
            )
            for i in range(1, len(slicing_indices))
        )
        for return_tuple in return_tuples:
            returned_max_eles, returned_max_belief, returned_beliefs, returned_num_idxs_with_belief_above_threshold, starting_index, ending_index = (
                return_tuple
            )
            # copy beliefs from results
            belief_space.beliefs[starting_index:ending_index] = returned_beliefs
            num_idxs_with_belief_above_threshold += (
                returned_num_idxs_with_belief_above_threshold
            )
            # find max chains
            if returned_max_belief > max_belief:
                max_eles = returned_max_eles
                max_belief = returned_max_belief
        # find actual max in tuple returned

        return max_eles, max_belief, num_idxs_with_belief_above_threshold


class BeliefSpace:
    def __init__(self, num_ele, belief_threshold=0.0, init_to_zero=False):
        if num_ele == 0:
            self.beliefs = None
        else:
            if init_to_zero:
                self.beliefs = np.zeros(num_ele)
            else:
                # initialize to uniform
                self.beliefs = np.full(num_ele, fill_value=1 / num_ele)
        self.belief_threshold = belief_threshold
        self.num_idxs_with_belief_above_threshold = num_ele

    def __len__(self):
        return len(self.beliefs)

    def __getitem__(self, item):
        return self.beliefs[item]

    def __setitem__(self, idx, value):
        self.beliefs[idx] = value

    def renormalize_beliefs(self, multiproc=False):
        """
        renormalizes beliefs to be a probablity distribution (normalizes belief between 0 and 1
        :param multiproc: whether or not to use multiprocessing
        :return: list of chains with the maximimal belief
        """

        self.num_idxs_with_belief_above_threshold = 0

        # compute normalization factor
        normalization_factor = self.sum_beliefs(multiproc=multiproc)
        assert (
            normalization_factor != 0
        ), "Normalization factor is zero: no chains with positive belief"

        if not multiproc:
            max_elements, max_belief, _, num_elements_with_belief_above_threshold, _, _ = self.renormalize_beliefs_common(
                normalization_factor, 0, len(self.beliefs)
            )
            self.num_idxs_with_belief_above_threshold = (
                num_elements_with_belief_above_threshold
            )
            return max_elements
        else:
            max_elements, max_belief, num_elements_with_belief_above_threshold = renormalize_beliefs_multiproc(
                self, normalization_factor
            )
            return max_elements

    def renormalize_beliefs_common(
        self, normalization_factor, starting_idx, ending_idx
    ):
        num_elements_with_belief_above_threshold = 0
        max_elements = []
        max_belief = -1
        for element_idx in range(starting_idx, ending_idx):
            element_belief = self.beliefs[element_idx]
            if element_belief <= 0:
                continue

            # if (
            #     causal_chain in self.causal_chain_space.true_chains
            #     and causal_chain.belief <= self.belief_threshold
            # ):
            #     self.print_message("TRUE GRAPH HAS BELIEF BELOW THRESHOLD:")
            #     self.causal_chain_space.pretty_print_causal_chains([causal_chain])

            # new way - belief is this chain's belief divided by normalization factor
            new_belief = element_belief / normalization_factor
            self.beliefs[element_idx] = new_belief

            if new_belief > self.belief_threshold:
                num_elements_with_belief_above_threshold += 1
            # if new_belief > 0.0:
            #     self.num_chains_with_positive_belief += 1

            if new_belief > max_belief:
                max_elements = [element_idx]
                max_belief = element_belief
            elif new_belief == max_belief:
                max_elements.append(element_idx)

        return (
            max_elements,
            max_belief,
            self.beliefs[starting_idx:ending_idx],
            num_elements_with_belief_above_threshold,
            starting_idx,
            ending_idx,
        )

    def compute_num_idx_with_belief_above_threshold(self, multiproc=True):

        if not multiproc:
            num_idxs_with_belief_above_threshold = self.compute_num_idxs_with_belief_above_threshold_common(
                0, len(self.beliefs)
            )
        else:
            num_idxs_with_belief_above_threshold = set_num_idxs_with_belief_above_threshold_multiproc(
                self
            )

        self.num_idxs_with_belief_above_threshold = num_idxs_with_belief_above_threshold
        print(
            "{} chains with belief above {}".format(
                self.num_idxs_with_belief_above_threshold, self.belief_threshold
            )
        )

    def compute_num_idxs_with_belief_above_threshold_common(
        self, starting_idx, ending_idx
    ):
        num_idxs_with_belief_above_threshold = sum(
            self.beliefs[i] > self.belief_threshold
            for i in range(starting_idx, ending_idx)
        )
        return int(num_idxs_with_belief_above_threshold)

    def verify_num_idxs_with_belief_above_threshold_is_correct(self):
        return self.num_idxs_with_belief_above_threshold == sum(
            self.beliefs[i] > self.belief_threshold for i in range(len(self.beliefs))
        )

    def verify_idxs_have_belief_above_threshold(self, idxs):
        idxs_with_belief_above_threshold = []
        for idx in idxs:
            if self.beliefs[idx] > self.belief_threshold:
                idxs_with_belief_above_threshold.append(idx)
        return len(idxs_with_belief_above_threshold) == len(idxs)

    def get_idxs_with_belief_above_threshold(self, print_msg=True):
        start_time = time.time()
        idxs_above_threshold = [
            i
            for i in range(len(self.beliefs))
            if self.beliefs[i] > self.belief_threshold
        ]
        if print_msg:
            print(
                "Getting {} indices with belief above threshold {} took {:0.6f}s".format(
                    len(idxs_above_threshold),
                    self.belief_threshold,
                    time.time() - start_time,
                )
            )
        if len(idxs_above_threshold) < SANITY_CHECK_ELEMENT_LIMIT:
            assert all(
                [
                    self.beliefs[idxs_above_threshold[i]] > 0
                    for i in range(len(idxs_above_threshold))
                ]
            )
        self.num_idxs_with_belief_above_threshold = len(idxs_above_threshold)
        return idxs_above_threshold

    def get_highest_N_belief_idxs(self, N):
        return get_highest_N_idxs(N, self.beliefs)

    def sum_beliefs(self, multiproc=False):
        if not multiproc:
            return self.sum_beliefs_common(0, len(self.beliefs))
        else:
            return sum_beliefs_multiproc(self)

    def sum_beliefs_common(self, starting_idx, ending_idx):
        return sum(
            self.beliefs[i]
            for i in range(starting_idx, ending_idx)
            if self.beliefs[i] > 0
        )

    def set_belief_threshold(
        self, set_threshold_from_num_ele=True, threshold_specified=None
    ):
        if threshold_specified is not None:
            self.belief_threshold = threshold_specified
        elif set_threshold_from_num_ele:
            uniform_belief = 1 / len(self.beliefs)
            self.belief_threshold = uniform_belief - uniform_belief / 10
        else:
            self.belief_threshold = 0
        self.belief_threshold = float(self.belief_threshold)

    # set uniform prior among all chains with a belief above a specified threshold
    def set_uniform_belief_for_ele_with_belief_above_threshold(
        self, threshold=0.0, multiproc=True
    ):
        start_time = time.time()
        print(
            "Setting uniform prior for indices with belief above {}...".format(
                threshold
            )
        )
        self.compute_num_idx_with_belief_above_threshold(multiproc=multiproc)
        assert_str = "No indices with belief above {}! Cannot set uniform belief".format(
            self.belief_threshold
        )
        assert self.num_idxs_with_belief_above_threshold, assert_str

        if not multiproc:
            self.beliefs, _, _ = self.set_uniform_belief_for_idx_with_belief_above_threshold_common(
                0, len(self.beliefs), threshold
            )
        else:
            self.beliefs = set_uniform_belief_for_idx_with_belief_above_threshold_multiproc(
                self, threshold
            )

        print(
            "Setting uniform belief for indices with belief above {} took {:0.6f}s".format(
                threshold, time.time() - start_time
            )
        )

    def set_uniform_belief_for_idx_with_belief_above_threshold_common(
        self, starting_idx, ending_idx, threshold=0.0
    ):
        for i in range(starting_idx, ending_idx):
            if self.beliefs[i] > threshold:
                self.beliefs[i] = 1 / self.num_idxs_with_belief_above_threshold

        return self.beliefs[starting_idx:ending_idx], starting_idx, ending_idx

    def set_uniform_belief(self):
        """
        sets uniform belief regardless of current belief value
        :param chain_list:
        :param reset_belief_counts:
        :return:
        """
        print("Setting uniform belief for all chains.")
        num_ele = len(self.beliefs)
        self.beliefs = np.full(num_ele, 1 / num_ele)

    def set_zero_belief(self):
        self.beliefs = np.zeros(len(self.beliefs))


class TopDownChainBeliefSpace(BeliefSpace):
    def __init__(self, num_ele, belief_threshold=0.0, init_to_zero=False):
        super(TopDownChainBeliefSpace, self).__init__(
            num_ele, belief_threshold, init_to_zero
        )

    def get_idxs_with_belief_above_threshold(self, print_msg=True):
        start_time = time.time()
        idxs_above_threshold = super(TopDownChainBeliefSpace, self).get_idxs_with_belief_above_threshold(print_msg=False)
        if print_msg:
            print(
                "Getting {} top-down indices with belief above threshold {} took {:0.6f}s".format(
                    len(idxs_above_threshold),
                    self.belief_threshold,
                    time.time() - start_time,
                )
            )
        return idxs_above_threshold


class BottomUpChainBeliefSpace(BeliefSpace):
    def __init__(
        self,
        num_ele,
        attribute_labels=None,
        belief_threshold=0.0,
        convert_to_ids=False,
        unique_id_manager=None,
        use_indexed_distributions=True,
        use_action_distribution=True,
    ):
        super(BottomUpChainBeliefSpace, self).__init__(num_ele, belief_threshold)
        self.use_indexed_distributions = use_indexed_distributions
        self.use_action_distribution = use_action_distribution
        if attribute_labels is not None:
            self.attribute_space = setup_attribute_space(
                attribute_labels,
                convert_to_ids=convert_to_ids,
                unique_id_manager=unique_id_manager,
            )

    def verify_true_chain_idxs_have_belief_above_threshold(self, idxs):
        return super(
            BottomUpChainBeliefSpace, self
        ).verify_idxs_have_belief_above_threshold(idxs)

    def get_idxs_with_belief_above_threshold(self, print_msg=True):
        start_time = time.time()
        idxs_above_threshold = super(BottomUpChainBeliefSpace, self).get_idxs_with_belief_above_threshold(print_msg=False)
        if print_msg:
            print(
                "Getting {} bottom-up indices with belief above threshold {} took {:0.6f}s".format(
                    len(idxs_above_threshold),
                    self.belief_threshold,
                    time.time() - start_time,
                )
            )
        return idxs_above_threshold


class AtomicSchemaBeliefSpace:
    def __init__(self, num_ele, belief_threshold=0.0, alpha_increase=1):
        self.schema_dirichlet = DirichletDistribution(num_ele)
        self.alpha_increase = alpha_increase

    def __getitem__(self, item):
        return self.beliefs[item]

    @property
    def beliefs(self):
        return self.schema_dirichlet.sampled_multinomial

    def update_alpha(self, solution_idx):
        self.schema_dirichlet.update_alpha(solution_idx, alpha_increase=self.alpha_increase)


class InstantiatedSchemaBeliefSpace(BeliefSpace):
    def __init__(self, num_ele, belief_threshold=0.0, init_to_zero=False):
        super(InstantiatedSchemaBeliefSpace, self).__init__(
            num_ele, belief_threshold, init_to_zero
        )

    def reset(self):
        self.beliefs = None


class AbstractSchemaBeliefSpace(BeliefSpace):
    def __init__(self, num_ele, belief_threshold=0.0, init_to_zero=False):
        super(AbstractSchemaBeliefSpace, self).__init__(num_ele, belief_threshold, init_to_zero)

    def reset(self):
        self.beliefs = None
