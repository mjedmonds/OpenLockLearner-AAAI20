import pickle
import glob


def write_causal_structure_space(
    causal_chain_structure_space, causal_chain_structure_space_path
):
    with open(causal_chain_structure_space_path, "wb") as causal_chain_space_file:
        pickle.dump(causal_chain_structure_space, causal_chain_space_file)
    print("Saved causal chain space to {}".format(causal_chain_structure_space_path))

def write_schema_structure_space(schema_structure_space, schema_structure_space_path):
    n_solutions = len(schema_structure_space.schemas[0].chains)
    with open(
        schema_structure_space_path, "wb"
    ) as schema_structure_space_file:
        pickle.dump(schema_structure_space, schema_structure_space_file)
    print(
        "Saved {} solution schemas to {}".format(n_solutions,
            schema_structure_space_path
        )
    )

def load_causal_structures_from_file(
    causal_chain_structure_space_path,
    two_solution_schemas_structure_space_path,
    three_solution_schemas_structure_space_path,
):
    with open(causal_chain_structure_space_path, "rb") as causal_chain_space_file:
        causal_chain_space = pickle.load(causal_chain_space_file)
    print("Loaded causal chain space from {}".format(causal_chain_structure_space_path))
    with open(
        two_solution_schemas_structure_space_path, "rb"
    ) as two_solution_schemas_file:
        two_solution_schemas = pickle.load(two_solution_schemas_file)
    print(
        "Loaded two solution schemas from {}".format(causal_chain_structure_space_path)
    )
    with open(
        three_solution_schemas_structure_space_path, "rb"
    ) as three_solution_schemas_file:
        three_solution_schemas = pickle.load(three_solution_schemas_file)
    print(
        "Loaded three solution schemas from {}".format(
            causal_chain_structure_space_path
        )
    )
    return causal_chain_space, two_solution_schemas, three_solution_schemas


