# cython: language_level=3

import itertools
import constraint
import numpy as np
import os

from openlockagents.OpenLockLearner.causal_classes.CausalRelation import (
    CausalRelationType,
    CausalRelation,
)
from openlock.common import Action

# import cython
# if cython.compiled:
#     print(os.path.basename(__file__) + " is compiled.")
# else:
#     print(os.path.basename(__file__) + " is an interpreted script.")


POSITIONS = [
    "UPPERRIGHT",
    "UPPER",
    "UPPERLEFT",
    "LEFT",
    "LOWERLEFT",
    "LOWER",
    "LOWERRIGHT",
]
COLORS = ["GREY", "WHITE"]
ACTIONS = ["push", "pull"]
# ACTIONS = ["push_{}".format(x) for x in POSITIONS] + ["pull_{}".format(POSITIONS) for x in POSITIONS]
FLUENT_STATES = [0, 1]
FLUENTS = [CausalRelationType.one_to_zero, CausalRelationType.zero_to_one]


class CausalRelationSpace:
    def __init__(
        self,
        actions,
        attributes,
        causal_relation_types,
        fluent_states,
        perceptually_causal_relations=None,
        true_chains=None,
    ):
        self.attributes = list(itertools.product(*attributes))
        self.actions = actions

        preconditions_with_dependencies = list(
            itertools.product(self.attributes, fluent_states)
        )
        self.causal_relation_types = causal_relation_types
        self.preconditions = [None]
        self.preconditions.extend(preconditions_with_dependencies)

        self.causal_relations_no_parent = list(
            itertools.product(
                [None], self.attributes, self.actions, self.causal_relation_types
            )
        )
        # self.causal_relations_parent = list(itertools.product(preconditions_with_dependencies, self.attributes, self.actions, self.fluents))

        # construct map of perceptually causal relations
        perceptually_causal_relation_map = dict()
        if perceptually_causal_relations is not None:
            for pcr in perceptually_causal_relations:
                perceptually_causal_relation_map[
                    (pcr.causal_relation.action, pcr.causal_relation.attributes)
                ] = pcr.causal_relation.causal_relation_type

        self.causal_relations = self.generate_causal_relations(
            preconditions=self.preconditions,
            attributes=self.attributes,
            actions=self.actions,
            causal_relation_types=self.causal_relation_types,
            perceptually_causal_relation_map=perceptually_causal_relation_map,
        )

        self.causal_relations_no_parent = set([
            x for x in self.causal_relations if x.precondition is None
        ])
        self.causal_relations_parent = set([
            x for x in self.causal_relations if x.precondition is not None
        ])

    @staticmethod
    def generate_causal_relations(
        preconditions,
        attributes,
        actions,
        causal_relation_types,
        perceptually_causal_relation_map,
    ):
        causal_relations = set()
        counter = 0
        for attribute in attributes:
            for action in actions:
                # todo: this is a hack but prevents a lot of problems later on
                # skip door pulling action
                if action == "pull" and attribute[0] == "door":
                    continue

                for precondition in preconditions:
                    for causal_relation_type in causal_relation_types:

                        # if any([CausalRelation(
                        #                         action=Action(action, attribute[0], None),
                        #                         attributes=attribute,
                        #                         causal_relation_type=causal_relation_type,
                        #                         precondition=precondition,
                        #                     ) in true_chain for true_chain in true_chains]):
                        #     print("true chain component")

                        # check against perceptually causal relations
                        if (action, attribute) in perceptually_causal_relation_map.keys():
                            # if the observed fluent change matches the generated one, we can add this relation
                            if (
                                perceptually_causal_relation_map[(action, attribute)]
                                == causal_relation_type
                            ):
                                causal_relations.add(
                                    (
                                        # todo: hacky way to add in position to action
                                        CausalRelation(
                                            action=Action(action, attribute[0], None),
                                            attributes=attribute,
                                            causal_relation_type=causal_relation_type,
                                            precondition=precondition,
                                        )
                                    )
                                )
                            # else:
                            #     print("Rejected: {}".format((action, attribute, causal_relation_type, precondition)))
                        else:
                            # this (action, attribute) pair is not in the perceptually causal relations, add this relation
                            causal_relations.add(
                                CausalRelation(
                                    # todo: hacky way to add in position to action
                                    action=Action(action, attribute[0], None),
                                    attributes=attribute,
                                    causal_relation_type=causal_relation_type,
                                    precondition=precondition,
                                )
                            )
                        counter += 1

        print(
            "{}/{} valid causal relations generated".format(
                len(causal_relations), counter
            )
        )
        return causal_relations

    @staticmethod
    def find_relations_satisfying_constraints(causal_relations, inclusion_constraints, exclusion_constraints=None):
        """
        Finds all causal relations adhere to the constraints specified in constraints_dict
        :param inclusion_constraints: a dictionary consistent of fields that make up the CausalRelation named tuple
        :return: a list of causal relations that satisfy the constraints
        """
        relations_satisfying_constraints = []

        for causal_relation in causal_relations:
            # relation not allowed in exclusion constraints
            if exclusion_constraints is not None and causal_relation in exclusion_constraints:
                continue
            constraints_satisfied = True
            # verify relation has the required constraints
            if inclusion_constraints is not None:
                for constraint_key, constraint_value in inclusion_constraints.items():
                    relation_value = getattr(causal_relation, constraint_key)
                    if relation_value != constraint_value:
                        constraints_satisfied = False
                        break
            if constraints_satisfied:
                relations_satisfying_constraints.append(causal_relation)

        return relations_satisfying_constraints

    @staticmethod
    def constraint_based_generation(
        preconditions,
        attributes,
        actions,
        causal_relation_types,
        perceptually_causal_relations,
    ):
        # manually generator the causal relations with parents so that the precondition matches the fluent precondition state
        precondition_match_csp = constraint.Problem()
        precondition_match_csp.addVariable("preconditions", preconditions)
        precondition_match_csp.addVariable("attributes", attributes)
        precondition_match_csp.addVariable("actions", actions)
        precondition_match_csp.addVariable("fluents", causal_relation_types)

        # add in constraint that precondition state must match fluent's precondition
        precondition_match_csp.addConstraint(
            lambda precondition, fluent: precondition is None
            or precondition[1] == fluent.value[0],
            ("preconditions", "fluents"),
        )

        # add constraints for perceptually causal relations
        for perceptually_causal_relation in perceptually_causal_relations:
            # causal relation (fluent change)
            precondition_match_csp.addConstraint(
                lambda attributes, action, fluent: fluent
                == perceptually_causal_relation.causal_relation_type
                if (
                    attributes == perceptually_causal_relation.attributes
                    and action == perceptually_causal_relation.action
                )
                else True,  # if the attributes and action do not match, always accept
                ("attributes", "actions", "fluents"),
            )
        causal_relations = precondition_match_csp.getSolutions()
        return causal_relations


if __name__ == "__main__":
    causal_relation_manager = CausalRelationSpace(
        ACTIONS, [POSITIONS, COLORS], FLUENT_STATES, FLUENTS
    )
