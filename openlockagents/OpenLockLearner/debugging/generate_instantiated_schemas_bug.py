import pickle as pkl

def main():
    with open("openlockagents/OpenLockLearner/debugging/schema_bug.pkl", "rb") as f:
        bug_data = pkl.load(f)
        abstract_schema, abstract_schema_belief, chain_assignments, causal_chain_structure_space, schemas, relations_used_in_instantiated_chains, excluded_chain_idxs, schema_structure_space = bug_data

        schema_structure_space.generate_instantiated_schemas(abstract_schema, abstract_schema_belief, chain_assignments, causal_chain_structure_space, schemas, relations_used_in_instantiated_chains, excluded_chain_idxs)

        print(bug_data)

if __name__ == "__main__":
    main()