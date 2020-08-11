import os

from openlockagents.OpenLockLearner.util.common import (
    CAUSAL_CHAIN_EDGES,
    FLUENTS,
    FLUENT_STATES,
    ACTIONS,
    setup_structure_space_paths
)
from openlockagents.common.agent import Agent
from openlockagents.OpenLockLearner.causal_classes.hypothesis_space import generate_hypothesis_space


def main():
    generate_causal_structures()


def generate_causal_structures():
    params = dict()
    params["use_physics"] = False
    params["train_scenario_name"] = 'CE3'
    params["src_dir"] = None

    env = Agent.pre_instantiation_setup(params, True)
    env.lever_index_mode = "position"

    attributes = [env.attribute_labels[attribute] for attribute in env.attribute_order]
    structure = CAUSAL_CHAIN_EDGES

    causal_chain_structure_space_path, two_solution_schemas_structure_space_path, three_solution_schemas_structure_space_path = setup_structure_space_paths()
    generate_hypothesis_space(env=env,
                              structure=structure,
                              causal_chain_structure_space_path=causal_chain_structure_space_path,
                              two_solution_schemas_structure_space_path=two_solution_schemas_structure_space_path,
                              three_solution_schemas_structure_space_path=three_solution_schemas_structure_space_path,
                              attributes=attributes,
                              actions=ACTIONS,
                              fluents=FLUENTS,
                              fluent_states=FLUENT_STATES,
                              perceptually_causal_relations=None)
    return


if __name__ == "__main__":
    main()