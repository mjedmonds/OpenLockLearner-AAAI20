

def setup_causal_chain_manager_params(causal_chain_space, initialization_size=1000):
    """
    setups params dictionary to initialize a CausalChainManagerAbstractClass derived class
    :param causal_chain_space: causal_chain_space to extract params from
    :param initialization_size: initial size of the CausalChainManagerAbstractClass derived class
    :return: a dictionary of params
    """
    params = dict()
    params["initialization_size"] = initialization_size
    params["num_states_in_chain"] = causal_chain_space.base_schema.num_states_in_chain
    params["num_attributes"] = causal_chain_space.num_attributes
    return params

