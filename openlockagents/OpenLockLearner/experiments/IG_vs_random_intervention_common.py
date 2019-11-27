
import gym

from openlockagents.OpenLockLearner.util.setup_util import setup_causal_chain_space, create_and_run_agent

def run_experiment(params, structure, perceptually_causal_relations, num_runs):
    # setup initial env
    env = gym.make("openlock-v1")
    env.use_physics = False
    env.initialize_for_scenario(params["train_scenario_name"])

    causal_chain_space = setup_causal_chain_space(
        env=env,
        structure=structure,
        perceptually_causal_relations=perceptually_causal_relations,
        multiproc=params["multiproc"],
        data_dir=params["chain_data_dir"],
        chain_mode=params["chain_mode"],
        prune_chain_space=params["prune_chain_space"],
        generate=params['generate_chains'],
        using_ids=params["using_ids"]
    )

    env.lever_index_mode = causal_chain_space.lever_index_mode

    random_num_chains_with_belief_above_threshold_per_attempt = []
    information_gain_num_chains_with_belief_above_threshold_per_attempt = []
    # run the same experiment multiple times
    for i in range(num_runs):
        params["use_random_intervention"] = True
        agent_random = create_and_run_agent(
            env,
            causal_chain_space,
            params,
            use_random_intervention=params["use_random_intervention"],
        )
        agent_random.finish_subject("random_intervention_selection", "random_intervention_selection")
        random_num_chains_with_belief_above_threshold_per_attempt.append(
            agent_random.num_chains_with_belief_above_threshold_per_attempt
        )
        causal_chain_space.reset()

        params["use_random_intervention"] = False
        agent_ig = create_and_run_agent(
            env,
            causal_chain_space,
            params,
            use_random_intervention=params["use_random_intervention"],
        )
        agent_ig.finish_subject("IG_intervention_selection", "IG_intervention_selection")
        information_gain_num_chains_with_belief_above_threshold_per_attempt.append(
            agent_ig.num_chains_with_belief_above_threshold_per_attempt
        )
        causal_chain_space.reset()