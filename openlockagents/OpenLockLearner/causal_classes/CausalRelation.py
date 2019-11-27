from enum import Enum
from collections import namedtuple

CausalRelation = namedtuple('CausalRelation', 'precondition action attributes causal_relation_type')


class CausalRelationType(Enum):
    zero_to_zero = (0,0)
    one_to_zero = (1,0)
    zero_to_one = (0,1)
    one_to_one = (1,1)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __getitem__(self, item):
        return self.value[item]

    def get_previous_state(self):
        return self.value[0]

    def get_current_state(self):
        return self.value[1]


class CausalRelationClass:
    def __init__(
        self, precondition, causal_relation_type, action, attributes
    ):
        self.precondition = precondition
        self.causal_relation_type = causal_relation_type
        self.action = action
        self.attributes = attributes

    def __eq__(self, other):
        return (
            self.precondition == other.precondition
            and self.causal_relation_type == other.causal_relation_type
            and self.action == other.action
            and self.attributes == other.attributes
        )

    def __str__(self):
        return "precondition: {}, causal_relation_type: {}, action: {}, attributes: {}".format(
            self.precondition, self.causal_relation_type, self.action, self.attributes
        )

    def __repr__(self):
        return "(" + str(self) + ")"

    def __hash__(self):
        return hash(
            (self.precondition, self.action, self.attributes, self.causal_relation_type)
        )


class CausalObservation:
    def __init__(self, causal_relation, info_gain=None):
        self.causal_relation = causal_relation

        if isinstance(info_gain, list):
            self.info_gains = info_gain
        else:
            self.info_gains = [info_gain] if info_gain is not None else []

    def add_info_gain(self, info_gain):
        self.info_gains.append(info_gain)

    def __str__(self):
        return str(self.causal_relation) + str(self.info_gains)

    def __repr__(self):
        return str(self)

    def determine_causal_change_occurred(self):
        return self.causal_relation.causal_relation_type is not None
