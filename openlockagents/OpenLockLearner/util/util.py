from openlockagents.OpenLockLearner.causal_classes.CausalChain import CausalChainCompact
from openlockagents.OpenLockLearner.causal_classes.CausalRelation import CausalRelation, CausalRelationType
from openlock.settings_scenario import select_scenario
from openlock.settings_trial import LEVER_CONFIGS

from openlockagents.OpenLockLearner.util.common import GRAPH_INT_TYPE

from openlock.common import ENTITY_STATES, Action



def generate_solutions_by_trial(scenario_name, trial_name):
    solution_chains = []
    scenario = select_scenario(scenario_name, use_physics=False)

    # todo: extract these from the environment/scenario somehow. these are hard-coded
    lever_cpt_choice = GRAPH_INT_TYPE(1)
    door_cpt_choice = GRAPH_INT_TYPE(0)
    lever_ending_state = GRAPH_INT_TYPE(ENTITY_STATES["LEVER_PUSHED"])
    door_ending_state = GRAPH_INT_TYPE(ENTITY_STATES["DOOR_OPENED"])

    scenario_solutions = scenario.solutions
    trial_levers = LEVER_CONFIGS[trial_name]
    for scenario_solution in scenario_solutions:
        solution_actions = []
        solution_states = []
        solution_cpt_choices = []
        solution_attributes = []
        solution_outcomes = []
        for action_log in scenario_solution:
            action_name = action_log.name
            state_name = action_name.split("_")[1]
            if state_name == "door":
                ending_state = door_ending_state
                cpt_choice = door_cpt_choice
            else:
                # determine position of lever based on role
                for trial_lever in trial_levers:
                    if trial_lever.LeverRoleEnum == state_name:
                        state_name = trial_lever.LeverPosition.name
                ending_state = lever_ending_state
                cpt_choice = lever_cpt_choice

            action_name = "push_" + state_name
            attributes = (state_name, "GREY")
            solution_actions.append(action_name)
            solution_states.append(state_name)
            solution_attributes.append(attributes)
            solution_cpt_choices.append(cpt_choice)
            solution_outcomes.append(ending_state)
        solution_chains.append(
            CausalChainCompact(
                states=tuple(solution_states),
                actions=tuple(solution_actions),
                conditional_probability_table_choices=tuple(solution_cpt_choices),
                outcomes=tuple(solution_outcomes),
                attributes=tuple(solution_attributes),
            )
        )
    return solution_chains


def generate_solutions_by_trial_causal_relation(scenario_name, trial_name):
    solution_chains = []
    scenario = select_scenario(scenario_name, use_physics=False)

    # todo: extract these from the environment/scenario somehow. these are hard-coded
    lever_causal_relation_type = CausalRelationType.one_to_zero
    door_causal_relation_type = CausalRelationType.zero_to_one

    scenario_solutions = scenario.solutions
    trial_levers = LEVER_CONFIGS[trial_name]
    for scenario_solution in scenario_solutions:
        solution_chain = []
        precondition = None
        for action_log in scenario_solution:
            action_name = action_log.name
            state_name = action_name.split("_")[1]
            if state_name == "door":
                causal_relation = door_causal_relation_type
            else:
                # determine position of lever based on role
                for trial_lever in trial_levers:
                    if trial_lever.LeverRoleEnum == state_name:
                        state_name = trial_lever.LeverPosition.name
                causal_relation = lever_causal_relation_type

            action_name = "push"
            attributes = (state_name, "GREY")
            solution_chain.append(
                CausalRelation(
                    action=Action(action_name, attributes[0], None),
                    attributes=attributes,
                    causal_relation_type=causal_relation,
                    precondition=precondition
                )
            )
            # setup precondition for next link in chain
            precondition = (attributes, causal_relation[1])
        solution_chains.append(tuple(solution_chain))
    return solution_chains
