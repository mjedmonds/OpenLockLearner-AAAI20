import numpy as np
import texttable

from openlockagents.OpenLockLearner.causal_classes.DirichletDistribution import (
    DirichletDistribution,
)


def setup_attribute_space(attributes, convert_to_ids=False, unique_id_manager=None):
    initial_attributes = list()
    for key, value in attributes.items():
        if convert_to_ids:
            attribute_labels = [
                unique_id_manager.convert_attribute_to_target_type(
                    key, value_label, target_type="int"
                )
                for value_label in value
            ]
        else:
            attribute_labels = value
        initial_attributes.append((key, len(attribute_labels), attribute_labels))
    attribute_space = AttributeSpace(
        initial_attributes=initial_attributes, using_ids=convert_to_ids
    )

    return attribute_space


class AttributeScope:
    """
    defines an Attribute scope, which manages a summary prior distribution over all causal change indices
    and the distributions over each causal change indices
    """

    def __init__(self, initial_attributes=None, prior=None):
        # type/sanity checking
        assert_str = "Expected a list of tuples, where first index in tuple is attribute name. Second index is attribute dimensionality. Third index is a list of labels for every dimension"
        assert isinstance(initial_attributes, list), assert_str
        for attribute in initial_attributes:
            assert isinstance(attribute, tuple), assert_str
            assert isinstance(attribute[0], str), assert_str
            assert isinstance(attribute[1], int), assert_str
            assert isinstance(attribute[2], list), assert_str
            assert attribute[1] == len(attribute[2]), assert_str

        names = []
        labels = []
        # initialize distributions to be uniform
        if initial_attributes is not None:
            names, dimensionalities, labels = zip(*initial_attributes)

        # summary distribution over all attributes
        self.summary_distributions = dict()
        # labels for each dimension of distribution
        self.labels = dict()
        # distributions based on effective causal change index; ie. how attributes change with changes in the
        self.indexed_distributions = []

        for i in range(len(names)):
            attribute_name = names[i]
            # start count at 1 (so this can be normalized into a distribution
            summary_attribute_prior = None
            if prior is not None:
                # initialize prior over summary distribution
                summary_attribute_prior = prior["summary"][attribute_name]

                # initialize prior at each indexed distribution
                for j in range(len(prior["indexed"])):
                    # add dict at this index if we don't already have one
                    if j >= len(self.indexed_distributions):
                        self.indexed_distributions.append(dict())
                    indexed_attribute_prior = prior["indexed"][j][attribute_name]
                    self.indexed_distributions[j][attribute_name] = DirichletDistribution(
                        dimensionalities[i], indexed_attribute_prior
                    )

            self.summary_distributions[attribute_name] = DirichletDistribution(
                dimensionalities[i], summary_attribute_prior
            )
            self.labels[attribute_name] = labels[i]

    def update_alpha(self, attribute_name, value, index, alpha_increase=1):
        """
        Adds a frequency of values to the name attribute.

        :param name: Name of attribute to add to
        :param values: vector of values to add to the existing frequency count
        :return: none
        """
        assert_str = "Adding indexed distribution at index {} when largest seen is {}. Max index possible is {}".format(
            index, len(self.indexed_distributions), len(self.indexed_distributions) + 1
        )
        assert index <= len(self.indexed_distributions), assert_str
        value_index = self.labels[attribute_name].index(value)
        self.summary_distributions[attribute_name].update_alpha(
            value_index, alpha_increase=alpha_increase
        )
        if index == len(self.indexed_distributions):
            # make a new distribution with the same dimensionality
            self.indexed_distributions.append(dict())
            for name in self.labels.keys():
                self.indexed_distributions[index][name] = DirichletDistribution(
                    len(self.summary_distributions[name].frequency_distribution)
                )

        self.indexed_distributions[index][attribute_name].update_alpha(
            value_index, alpha_increase=alpha_increase
        )

    def compute_chain_posterior(
        self, attribute_order, attributes_at_indices, actions_at_indices, use_confidence=False, use_indexed_distributions=True, use_action_distribution=True,
    ):
        posterior_at_indices = np.zeros((len(attributes_at_indices), 1))
        num_posteriors_computed = 0

        # use index distributions for as many as possible, then use summary distribution to estimate posterior of remaining indices
        distributions = []
        if use_indexed_distributions:
            distributions.extend([self.indexed_distributions[i] for i in range(len(self.indexed_distributions))])
        num_distributions = len(distributions)
        distributions.extend([self.summary_distributions for i in range(num_distributions, len(attributes_at_indices))])

        # compute node posteriors
        for i in range(len(distributions)):
            posterior_at_indices[i] = self._compute_node_posterior(
                distributions[i],
                attribute_order,
                attributes_at_indices[i],
                actions_at_indices[i],
                use_confidence,
                use_action_distribution,
            )
            num_posteriors_computed += 1

        # final posterior is product over each index (this should be normalized because the products are normalized and independent
        posterior = np.prod(posterior_at_indices)
        # assert (
        #     posterior != 0
        # ), "Posterior of chain is zero - but chain has not been pruned"
        return posterior


    def _compute_node_posterior(
        self, dist_dict, attribute_order, attributes, action, use_confidence=False, use_action=True
    ):
        attribute_value_idxs = [
            self.labels[attribute_order[j]].index(attributes[j])
            for j in range(len(attribute_order))
        ]
        node_posterior = 1
        assert len(attribute_order) == len(
            attribute_value_idxs
        ), "Should have same number of attributes as attribute values"
        for i in range(len(attribute_order)):
            attribute = attribute_order[i]
            # shouldn't these sample the dirichlet to get a multinomial?

            # two sampling options: from multinomial sampled from dirichlet or from multinormial frequency distribution
            # use sampled multinomial from Dirichlet (updated every renormalize() call
            dist = dist_dict[attribute].sampled_multinomial
            # directly use the frequency distribution
            # dist = dist_dict[attribute].frequency_distribution

            attribute_belief = dist[attribute_value_idxs[i]]
            # if attribute_belief <= 0:
            #     print("belief is <= 0")
            #     raise ValueError("attribute belief has 0 probability")
            node_posterior *= attribute_belief

        # add in the action belief
        if use_action:
            action_dist = dist_dict["action"].sampled_multinomial
            action_idx = self.labels["action"].index(action.name)
            node_posterior *= action_dist[action_idx]

        # these two lines compute full product, this normalizes to 1
        # attr_product = np.outer(color_dist, position_dist)
        # node_posterior = attr_product[color_idx][position_idx]
        return node_posterior

    def get_distribution_at_index(self, index):
        if index in range(len(self.indexed_distributions)):
            return self.indexed_distributions[index]
        else:
            return self.summary_distributions

    def get_attributes_info(self):
        attributes_info = []
        for name in self.labels.keys():
            dimensionality = self.summary_distributions[
                name
            ].frequency_distribution.shape[0]
            labels = self.labels[name]
            attributes_info.append((name, dimensionality, labels))
        return attributes_info

    def convert_to_list(self):
        for indexed_dist in self.indexed_distributions:
            for dist in indexed_dist.values():
                dist.convert_to_list()
        for dist in self.summary_distributions.values():
            dist.convert_to_list()

    def pretty_print(self):
        table = texttable.Texttable()
        attribute_order = list(self.summary_distributions.keys())

        distribution_content = []
        new_content = ["summary"]
        new_content.extend(
            [
                self.summary_distributions[attribute_order[i]].pretty_distribution()
                for i in range(len(attribute_order))
            ]
        )
        distribution_content.append(new_content)

        for j in range(len(self.indexed_distributions)):
            new_content = ["subchain" + str(j)]
            new_content.extend(
                [
                    self.indexed_distributions[j][
                        attribute_order[i]
                    ].pretty_distribution()
                    for i in range(len(attribute_order))
                ]
            )
            distribution_content.append(new_content)

        headers = ["dist label/index"]
        headers.extend(
            [
                attribute_order[i] + "\n" + str(self.labels[attribute_order[i]])
                for i in range(len(attribute_order))
            ]
        )
        alignment = ["c" for i in range(len(headers))]
        table.set_cols_align(alignment)
        content = [headers]
        content.extend(distribution_content)

        table.add_rows(content)

        widths = [20]
        widths.extend([100 for i in range(len(attribute_order))])
        table.set_cols_width(widths)

        print(table.draw())


class AttributeSpace:
    def __init__(
        self,
        initial_attributes=None,
        using_ids=True,
        global_alpha_update=1,
        local_alpha_update=1,
    ):
        # global attributes are summaries across attributes
        self.global_attributes = AttributeScope(initial_attributes)
        # local attributes are specific to each trial
        self.local_attributes = dict()
        self.global_alpha_update = global_alpha_update
        self.local_alpha_update = local_alpha_update
        self.using_ids = using_ids

    def add_frequency(
        self,
        attribute_name,
        value,
        trial_name,
        index,
        global_alpha_increase=1,
        local_alpha_increase=1,
    ):
        """
        Adds a frequency to the global attribute and the the corresponding local attribute
        :param attribute_name: Name of the attribute to add a frequency to
        :param value: Label of the value to add a frequency to
        :param trial_name: Trial, used for local attribute space
        :param index: causal change index, used to update index-specific attributes
        :return:
        """
        assert_str = "Trial {} expected in local attributes".format(trial_name)
        assert trial_name in self.local_attributes.keys(), assert_str

        self.global_attributes.update_alpha(
            attribute_name, value, index, alpha_increase=global_alpha_increase
        )
        self.local_attributes[trial_name].update_alpha(
            attribute_name, value, index, alpha_increase=local_alpha_increase
        )

    def initialize_local_attributes(
        self, trial_name, use_scaled_prior=False, scale_min=1, scale_max=10
    ):
        if len(self.local_attributes) == 0:
            self.local_attributes[trial_name] = AttributeScope(
                self.global_attributes.get_attributes_info()
            )
        else:
            if trial_name not in self.local_attributes.keys():
                # create new local attributes from prior over global attributes
                # self.local_attributes[trial_name] = AttributeScope(
                #     self.global_attributes.get_attributes_info(),
                #     prior=self.create_prior(self.global_attributes),
                # )
                self.local_attributes[trial_name] = AttributeScope(
                    self.global_attributes.get_attributes_info(),
                    prior=create_prior(
                        self.global_attributes,
                        scaled=use_scaled_prior,
                        scale_min=scale_min,
                        scale_max=scale_max,
                    ),
                )

    def add_frequencies(
        self,
        attribute_value_pairs,
        trial,
        index,
        global_alpha_increase=1,
        local_alpha_increase=1,
    ):
        for attribute_name, value in attribute_value_pairs:
            self.add_frequency(
                attribute_name,
                value,
                trial,
                index,
                global_alpha_increase=global_alpha_increase,
                local_alpha_increase=local_alpha_increase,
            )

    @staticmethod
    def create_sub_prior(dist_dict, **kwargs):
        prior = dict()
        for key in dist_dict.keys():
            prior[key] = dist_dict[key].sample_multinomial()
        return prior

    @staticmethod
    def create_scaled_sub_prior(dist_dict, **kwargs):
        scale_min = kwargs["scale_min"]
        scale_max = kwargs["scale_max"]
        prior = dict()
        for key in dist_dict.keys():
            prior[key] = AttributeSpace.scale_array(
                dist_dict[key].sample_multinomial(), scale_min, scale_max
            )
        return prior

    @staticmethod
    def scale_array(arr, scale_min, scale_max):
        return (
            (arr - arr.min()) * (1 / (arr.max() - arr.min())) * (scale_max - scale_min)
        ) + scale_min

    def convert_to_list(self):
        self.global_attributes.convert_to_list()

        for trial_scope in self.local_attributes.values():
            trial_scope.convert_to_list()

    def pretty_print_global_attributes(self):
        print("GLOBAL ATTRIBUTE BELIEFS")
        self.global_attributes.pretty_print()

    def pretty_print_local_attributes(self, trial_name):
        print("LOCAL ATTRIBUTE BELIEFS for {}".format(trial_name))
        self.local_attributes[trial_name].pretty_print()


def create_prior(attribute_scope, scaled=False, scale_min=1, scale_max=10):
    if scaled:
        prior_gen_func = AttributeSpace.create_scaled_sub_prior
    else:
        prior_gen_func = AttributeSpace.create_sub_prior
    prior = dict()
    prior["summary"] = prior_gen_func(
        attribute_scope.summary_distributions,
        **{"scale_min": scale_min, "scale_max": scale_max}
    )
    indexed_prior = []
    for i in range(len(attribute_scope.indexed_distributions)):
        indexed_prior.append(
            prior_gen_func(
                attribute_scope.indexed_distributions[i],
                **{"scale_min": scale_min, "scale_max": scale_max}
            )
        )
    prior["indexed"] = indexed_prior
    return prior
