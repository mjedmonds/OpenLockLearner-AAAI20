

import dill

from openlockagents.OpenLockLearner.learner.ModelBasedRL import ModelBasedRLAgent

if __name__ == '__main__':
    with open("/home/mark/Desktop/greedy_action_policy_problem_ce3_cc4.dill", 'rb') as f:
        variables = dill.load(f)

    model_based_agent = variables['model_based_agent']
    causal_chain_space = variables["causal_chain_space"]
    causal_chain_idxs = variables["causal_chain_idxs"]
    causal_change_idx = variables["causal_change_idx"]
    action_sequence = variables["action_sequence"]
    first_agent_trial = variables["first_agent_trial"]
    intervention_idxs_executed = variables["intervention_idxs_executed"]
    interventions_executed = variables["interventions_executed"]
    ablation = variables["ablation"]

    action, action_beliefs = model_based_agent.greedy_action_policy(
        causal_chain_space,
        causal_chain_idxs,
        causal_change_idx,
        action_sequence,
        first_agent_trial,
        intervention_idxs_executed,
        interventions_executed,
        ablation,
    )

    print(action_beliefs)
