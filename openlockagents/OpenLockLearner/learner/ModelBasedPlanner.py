

class ModelBasedPlanner:
    def __init__(self, goal):
        """
        initialize model based planner with a goal, where the goal represents a list of tuples (state, value) that expects the state to be value for the goal to be achieved
        :param goal: list of tuples, each of which represents a state and a value of the goal to be achieved (state, value)
        """
        self.goal = None
        self.goal_state_value_dict = dict()
        self.num_goals_to_satisfy = 0
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = goal
        goal_states, goal_values = zip(*self.goal)
        self.goal_state_value_dict = dict()
        for i in range(len(goal_states)):
            # map between goal states and a [goal_value, goal_satisfied] list
            self.goal_state_value_dict[goal_states[i]] = goal_values[i]

        self.num_goals_to_satisfy = len(goal_states)

    def find_chains_with_goal(self, causal_chain_structure_space, causal_chain_idxs):
        assert self.goal is not None, "ModelBasedPlanner has no goal set"

        chain_idxs_satisfying_goal = []
        for causal_chain_idx in causal_chain_idxs:
            # chain satisfied all goals
            if self.determine_chain_satisfies_goal(causal_chain_structure_space, causal_chain_idx):
                chain_idxs_satisfying_goal.append(causal_chain_idx)

        return chain_idxs_satisfying_goal

    def determine_chain_satisfies_goal(self, causal_chain_structure_space, causal_chain_idx):
        chain_chain = causal_chain_structure_space.causal_chains[causal_chain_idx]
        num_goals_satisfied = 0
        for i in range(len(chain_chain)):
            chain_position = chain_chain[i].attributes[causal_chain_structure_space.state_index_in_attributes]
            chain_outcome = chain_chain[i].causal_relation_type[1]
            try:
                goal_value = self.goal_state_value_dict[chain_position]
                if chain_outcome == goal_value:
                    num_goals_satisfied += 1
            except KeyError:
                continue

        # chain satisfied all goals
        return num_goals_satisfied == self.num_goals_to_satisfy

