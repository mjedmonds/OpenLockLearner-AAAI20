from openlockagents.OpenLockLearner.util.common import (
    GRAPH_INT_TYPE,
    homogenous_list,
    create_birdirectional_dict,
)
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalObservation,
)

class UniqueIDManager:

    def __init__(self, states, actions, attributes, attribute_order):
        self.attribute_order = attribute_order

        # instantiate bidirectional hash tables to map compact chains to integers
        self.state_to_state_id_bidir_dict = create_birdirectional_dict(
            states, GRAPH_INT_TYPE
        )
        self.action_to_action_id_bidir_dict = create_birdirectional_dict(
            actions, GRAPH_INT_TYPE
        )
        self.attribute_to_attribute_id_bidir_dict = dict()
        for i in range(len(self.attribute_order)):
            attribute = self.attribute_order[i]
            self.attribute_to_attribute_id_bidir_dict[
                attribute
            ] = create_birdirectional_dict(attributes[i], GRAPH_INT_TYPE)

    def convert_chain_to_target_type(self, causal_chain, target_type):
        new_states, new_actions, new_attributes = self.convert_states_actions_attributes_to_target_type(
            target_type,
            causal_chain.states,
            causal_chain.actions,
            causal_chain.attributes,
        )
        causal_chain.states = new_states
        causal_chain.actions = new_actions
        causal_chain.attributes = new_attributes
        return causal_chain

    def convert_perceptually_causal_relations_to_target_type(
        self, perceptually_causal_relations, target_type
    ):
        for perceptually_causal_relation in perceptually_causal_relations:
            # skip any states we are not considering as part of our state space
            if (
                perceptually_causal_relation.state
                not in self.state_to_state_id_bidir_dict.keys()
            ):
                continue
            perceptually_causal_relation.state = self.convert_state_to_target_type(
                perceptually_causal_relation.state, target_type
            )
            perceptually_causal_relation.action = self.convert_action_to_target_type(
                perceptually_causal_relation.action, target_type
            )
            # todo: this only does one attribute
            perceptually_causal_relation.attributes = self.convert_attribute_tuple_to_target_type(
                perceptually_causal_relation.attributes, target_type
            )
        return perceptually_causal_relations

    def convert_item_to_target_type(self, item, bidirectional_dict, target_type):
        if target_type == "str" and isinstance(item, GRAPH_INT_TYPE):
            item = bidirectional_dict[item]
        elif target_type == "int" and isinstance(item, str):
            item = bidirectional_dict[item]
        return item

    def convert_action_to_target_type(self, action, target_type):
        return self.convert_item_to_target_type(
            action, self.action_to_action_id_bidir_dict, target_type
        )

    def convert_state_to_target_type(self, state, target_type):
        return self.convert_item_to_target_type(
            state, self.state_to_state_id_bidir_dict, target_type
        )

    def convert_attribute_to_target_type(self, attribute, value, target_type):
        return self.convert_item_to_target_type(
            value, self.attribute_to_attribute_id_bidir_dict[attribute], target_type
        )

    def convert_items_using_bidirectional_dict(self, items_, bidirectional_dict):
        """
        converts back and forth between entries in the bidirectional hashtable dict_
        """
        items_other = tuple([bidirectional_dict[i] for i in items_])
        return items_other

    def convert_list_to_target_type(self, seq, conversion_dict, target_type):
        if target_type == "int":
            # verify all are strs before we convert
            if homogenous_list(seq) and isinstance(seq[0], str):
                return self.convert_items_using_bidirectional_dict(seq, conversion_dict)
        elif target_type == "str":
            # verify all are ints before we convert
            if homogenous_list(seq) and isinstance(seq[0], GRAPH_INT_TYPE):
                return self.convert_items_using_bidirectional_dict(seq, conversion_dict)
        else:
            raise ValueError("Expected conversion with target type of 'int' or 'str'")
        # no conversion necessary, already in target type
        return seq

    def convert_actions_to_target_type(self, actions, target_type):
        return self.convert_list_to_target_type(
            actions, self.action_to_action_id_bidir_dict, target_type
        )

    def convert_states_to_target_type(self, states, target_type):
        return self.convert_list_to_target_type(
            states, self.state_to_state_id_bidir_dict, target_type
        )

    def convert_attribute_tuple_to_target_type(self, attribute_values, target_type):
        """
        Converts a tuple of attributes to the specified target type
        :param attribute_values:
        :param target_type:
        :return:
        """
        new_attribute_values = []
        for i in range(len(self.attribute_order)):
            attribute = self.attribute_order[i]
            new_attribute_values.append(
                self.convert_item_to_target_type(
                    attribute_values[i],
                    self.attribute_to_attribute_id_bidir_dict[attribute],
                    target_type,
                )
            )
        return tuple(new_attribute_values)

    def convert_attribute_list_of_tuples_to_target_type(
        self, attribute_list, target_type
    ):
        """
        Converts a list of tuples of attributes into the specified target type
        :param attribute_list:
        :param target_type:
        :return:
        """
        return tuple(
            [
                self.convert_attribute_tuple_to_target_type(
                    attribute_tuple, target_type
                )
                for attribute_tuple in attribute_list
            ]
        )

    def convert_states_actions_attributes_to_target_type(
        self, target_type, states=None, actions=None, attributes=None
    ):
        states_other = None
        actions_other = None
        attributes_other = None
        if states is not None:
            states_other = self.convert_states_to_target_type(states, target_type)
        if actions is not None:
            actions_other = self.convert_actions_to_target_type(actions, target_type)
        if attributes is not None:
            attributes_other = self.convert_attribute_list_of_tuples_to_target_type(
                attributes, target_type
            )
        return states_other, actions_other, attributes_other

    def convert_causal_observation_to_target_type(
        self, causal_observation, target_type
    ):
        new_attributes = None
        new_state = None
        new_action = None
        if causal_observation.action is not None:
            new_action = self.convert_action_to_target_type(
                causal_observation.action, target_type
            )
        if causal_observation.state is not None:
            new_state = self.convert_state_to_target_type(
                causal_observation.state, target_type
            )
        if causal_observation.attributes is not None:
            new_attributes = tuple(
                self.convert_attribute_tuple_to_target_type(
                    causal_observation.attributes, target_type
                )
            )
        new_observation = CausalObservation(
            new_state,
            causal_observation.causal_relation_type,
            new_action,
            causal_observation.info_gains,
            new_attributes,
        )
        return new_observation
