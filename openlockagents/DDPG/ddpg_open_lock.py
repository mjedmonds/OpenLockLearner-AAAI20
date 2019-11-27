# -*- coding: utf-8 -*-
import gym
import os
import sys
import json

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from openlockagents.agent import Agent, ROOT_DIR

# MUST IMPORT FROM openlock to properly register the environment
from openlock.settings_trial import PARAMS, IDX_TO_PARAMS
from openlock.settings_scenario import select_scenario
from openlock.envs.openlock_env import ObservationSpace
from openlockagents.DDPG.ddpg_agent import DDPGAgent


EPISODES = 1000


def run_trials(
    agent,
    trial_count,
    num_iters,
    num_trials,
    scenario_name,
    action_limit,
    attempt_limit,
    use_dynamic_epsilon,
    dynamic_epsilon_max,
    dynamic_epsilon_decay,
    test_trial,
    fig=None,
):
    # train over multiple iterations over all trials
    for iter_num in range(num_iters):
        agent.env.completed_trials = []
        for trial_num in range(0, num_trials):
            agent = run_single_trial(
                agent,
                trial_num,
                iter_num,
                scenario_name,
                action_limit,
                attempt_limit,
                use_dynamic_epsilon,
                dynamic_epsilon_max,
                dynamic_epsilon_decay,
                test_trial,
                fig=fig,
            )
            trial_count += 1

    return agent, trial_count


def run_single_trial(
    agent,
    trial_num,
    iter_num,
    scenario_name,
    action_limit,
    attempt_limit,
    use_dynamic_epsilon=False,
    dynamic_max=None,
    dynamic_decay=None,
    test_trial=False,
    fig=None,
):
    agent.run_trial_ddpg(
        scenario_name=scenario_name,
        action_limit=action_limit,
        attempt_limit=attempt_limit,
        trial_count=trial_num,
        iter_num=iter_num,
        test_trial=test_trial,
        fig=fig,
    )
    agent.plot_rewards(
        agent.rewards,
        agent.epsilons,
        agent.writer.subject_path + "/training_rewards.png",
    )
    print(
        "Trial complete for subject {}. Average reward: {}".format(
            agent.logger.subject_id, agent.average_trial_rewards[-1]
        )
    )
    # reset the epsilon after each trial (to allow more exploration)
    return agent


# trains the transfer case and trains multiple transfer cases
def train_transfer_test_transfer(agent, fig=None):
    # train all training cases/trials
    params = agent.params
    trial_count = 0
    agent, trial_count = run_trials(
        agent,
        trial_count,
        params["train_num_iters"],
        params["train_num_trials"],
        params["train_scenario_name"],
        params["train_action_limit"],
        params["train_attempt_limit"],
        params["use_dynamic_epsilon"],
        params["dynamic_epsilon_max"],
        params["dynamic_epsilon_decay"],
        test_trial=False,
        fig=fig,
    )

    agent.plot_rewards(
        agent.rewards,
        agent.epsilons,
        agent.writer.subject_path + "/training_rewards.png",
    )
    agent.plot_rewards_trial_switch_points(
        agent.rewards,
        agent.epsilons,
        agent.trial_switch_points,
        agent.writer.subject_path + "/training_rewards_switch_points.png",
        plot_xticks=False,
    )
    agent.test_start_reward_idx = len(agent.rewards)
    agent.test_start_trial_count = trial_count

    agent.save_weights(
        agent.writer.subject_path + "/models", "/training_final.cpkt", sess=agent.sess
    )

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    if params["test_scenario_name"] is not None:

        # setup testing trial
        scenario = select_scenario(
            params["test_scenario_name"], use_physics=params["use_physics"]
        )
        agent.env.update_scenario(scenario)
        agent.env.set_action_limit(params["test_action_limit"])
        agent.env.observation_space = ObservationSpace(
            len(scenario.levers), append_solutions_remaining=False
        )

        agent, trial_count = run_trials(
            agent,
            trial_count,
            params["test_num_iters"],
            params["test_num_trials"],
            params["test_scenario_name"],
            params["test_action_limit"],
            params["test_attempt_limit"],
            params["use_dynamic_epsilon"],
            params["dynamic_epsilon_max"],
            params["dynamic_epsilon_decay"],
            test_trial=True,
        )

        agent.plot_rewards(
            agent.rewards[agent.test_start_reward_idx :],
            agent.epsilons[agent.test_start_reward_idx :],
            agent.writer.subject_path + "/testing_rewards.png",
            width=6,
            height=6,
        )
        agent.save_weights(agent.writer.subject_path + "/models", "/testing_final.h5")

    return agent


def train_single_trial(agent, params, fig=None):
    agent = run_single_trial(
        agent,
        trial_num=0,
        iter_num=0,
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
        fig=fig,
    )
    agent.plot_rewards(
        agent.rewards,
        agent.epsilons,
        agent.writer.subject_path + "/training_rewards.png",
    )
    agent.plot_rewards_trial_switch_points(
        agent.rewards,
        agent.epsilons,
        agent.trial_switch_points,
        agent.writer.subject_path + "/training_rewards_switch_points.png",
        plot_xticks=False,
    )
    agent.save_model(agent.writer.subject_path + "/models", "/training_final.h5")
    return agent


def create_reward_fig():
    # creating the figure
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    plt.ion()
    plt.show()
    return fig


def main():
    # general params
    # training params
    if len(sys.argv) < 2:
        # general params
        # training params
        # PICK ONE and comment others
        params = PARAMS["CE3-CE4"]
        # params = PARAMS['CE3-CC4']
        # params = PARAMS['CC3-CE4']
        # params = PARAMS['CC3-CC4']
        # params = PARAMS['CE4']
        # params = PARAMS['CC4']
    else:
        setting = sys.argv[1]
        params = PARAMS[IDX_TO_PARAMS[int(setting) - 1]]
        print(
            "training_scenario: {}, testing_scenario: {}".format(
                params["train_scenario_name"], params["test_scenario_name"]
            )
        )
        params["reward_mode"] = sys.argv[2]

    human_decay_mean = 0.7429  # from human data
    human_decay_median = 0.5480  # from human data

    # RL specific settings
    random_seed = 1234
    params["use_physics"] = False
    params[
        "full_attempt_limit"
    ] = (
        True
    )  # run to the full attempt limit, regardless of whether or not all solutions were found
    params["train_num_iters"] = 100
    params["test_num_iters"] = 10
    # params['epsilon_decay'] = 0.9955
    # params['epsilon_decay'] = 0.9999
    params["epsilon_decay"] = 0.99999
    params["dynamic_epsilon_decay"] = 0.9955
    params["dynamic_epsilon_max"] = 0.5
    params["use_dynamic_epsilon"] = True
    params["test_num_trials"] = 5

    params["data_dir"] = os.path.dirname(ROOT_DIR) + "/OpenLockRLResults/subjects"
    params["train_attempt_limit"] = 300
    params["test_attempt_limit"] = 300
    params["gamma"] = 0.8  # discount rate
    params["epsilon"] = 1.0  # exploration rate
    params["epsilon_min"] = 0.00
    params["learning_rate"] = 0.0005
    params["batch_size"] = 64

    # SINGLE TRIAL TRAINING
    # params['train_attempt_limit'] = 30000
    # params['epsilon_decay'] = 0.99995
    # params['use_dynamic_epsilon'] = False

    # dummy settings
    # params['train_num_iters'] = 10
    # params['test_num_iters'] = 10
    # params['train_attempt_limit'] = 30
    # params['test_attempt_limit'] = 30

    # human comparison settings
    # params['train_num_iters'] = 1
    # params['test_num_iters'] = 1
    # params['train_attempt_limit'] = 300000
    # params['test_attempt_limit'] = 300000
    # params['epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_decay'] = human_decay_mean
    # params['dynamic_epsilon_max'] = 1
    # params['use_dynamic_epsilon'] = True

    scenario = select_scenario(
        params["train_scenario_name"], use_physics=params["use_physics"]
    )

    # setup initial env
    env = gym.make("openlock-v1")

    env.use_physics = params["use_physics"]
    env.full_attempt_limit = params["full_attempt_limit"]

    # set up observation space
    env.observation_space = ObservationSpace(
        len(scenario.levers), append_solutions_remaining=False
    )

    # set reward mode
    env.reward_mode = params["reward_mode"]
    print("Reward mode: {}".format(env.reward_mode))

    agent = DDPGAgent(env, 1, 1, params, None, "init")

    # create session/trial/experiment
    # TODO: passing a fake agent here is a hack
    np.random.seed(random_seed)
    env.seed(random_seed)
    trial_selected = agent.setup_trial(
        scenario_name=params["train_scenario_name"],
        action_limit=params["train_action_limit"],
        attempt_limit=params["train_attempt_limit"],
    )

    env.reset()

    # setup agent
    state_size = agent.env.observation_space.multi_discrete.shape[0]
    action_size = len(agent.env.action_space)

    # agent = DQNAgent(state_size, action_size, params)

    sess = tf.Session()
    agent = DDPGAgent(env, state_size, action_size, params, sess, "DDPG")
    # update agent to be a properly initialized agent

    agent.env.reset()
    fig = create_reward_fig()
    agent.sess.run(tf.global_variables_initializer())

    # MULTI-TRIAL TRAINING, TESTING
    # runs through all training trials and testing trials
    agent = train_transfer_test_transfer(agent, fig)

    # SINGLE TRIAL TRAINING
    # agent, env, agent = train_single_trial(agent, env, agent, params, fig)

    agent.finish_subject()
    print("Training & testing complete for subject {}".format(agent.logger.subject_id))


def replot_training_results(path):
    agent_json = json.load(open(path))
    agent_folder = os.path.dirname(path)
    Agent.plot_rewards(
        agent_json["rewards"], agent_json["epsilons"], agent_folder + "/reward_plot.png"
    )


if __name__ == "__main__":
    # agent_path = '../OpenLockRLResults/negative_immovable_partial_seq/2014838386/2014838386_agent.json'
    # replot_training_results(agent_path)
    main()
