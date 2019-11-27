import time
import texttable
import sys
import os
import copy
import atexit
import json
import jsonpickle
import random
import gym
import pprint

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from openlockagents.logger_agent import SubjectLogger, SubjectWriter
from openlock.settings_trial import select_trial, get_trial, get_possible_trials
import openlock.common as common
from openlock.envs.openlock_env import ObservationSpace
from openlockagents.common import ROOT_DIR, DEBUGGING
from openlockagents.OpenLockLearner.util.common import write_source_code


def load_agent(subject_path):
    agent_id = os.path.basename(subject_path)
    with open(subject_path + "/" + agent_id + "_agent.json") as agent_file:
        json_str = agent_file.read()
        agent = jsonpickle.loads(json_str)
        return agent


# base class for all agents; each agent has a logger
class Agent(object):
    """
    Manage the agent's internals (e.g. neural network for DQN/DDQN, or q-table for Q-table)
    and maintain a logger to record the outcomes the agent achieves.
    """

    def __init__(self, name, params, env=None, random_seed=None):
        """
        Initialize logger, writer, subject_id, human to None; data_path to data_path.

        :param data_path: path to directory to write log files to
        """
        self.name = name
        self.logger = None
        self.writer = None
        self.subject_id = None
        self.data_path = os.path.expanduser(params["data_dir"])
        self.human = False
        self.params = params
        self.env = env
        self.random_seed = random_seed

        self.finished = False

        self.total_attempt_count = 0

        # register exit handler to finish the agent
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.env is not None and self.env.use_physics:
            self.env.render(self.env, close=True)  # close the window
        if self.writer is not None:
            self.writer.terminate()
        if not self.finished and self.logger is not None:
            self.finish_subject(
                strategy="EARLY TERMINATION", transfer_strategy="EARLY TERMINATION"
            )

    # default args are for non-human agent
    def setup_subject(
        self,
        human=False,
        participant_id=-1,
        age=-1,
        gender="robot",
        handedness="none",
        eyewear="no",
        major="robotics",
        random_seed=None,
        project_src=None,
    ):
        """
        Set internal variables for subject, initialize logger, and create a copy of the code base for reproduction.

        :param human: True if human agent, default: False
        :param participant_id: default: -1
        :param age: default: -1
        :param gender: default: 'robot'
        :param handedness: default: 'none'
        :param eyewear: default: 'no'
        :param major: default: 'robotics'
        :param random_seed: default: None
        :return: Nothing
        """
        assert project_src is not None, "Must specify a root directory for project source code"
        self.human = human
        self.writer = SubjectWriter(self.data_path)
        self.subject_id = self.writer.subject_id

        # redirect stdout to logger
        sys.stdout = self.writer.terminal_writer

        print(
            "Starting trials for subject {}. Saving to {}".format(
                self.subject_id, self.writer.subject_path
            )
        )
        self.logger = SubjectLogger(
            subject_id=self.subject_id,
            participant_id=participant_id,
            age=age,
            gender=gender,
            handedness=handedness,
            eyewear=eyewear,
            major=major,
            start_time=time.time(),
            random_seed=random_seed,
        )

        # copy the entire code base; this is unnecessary but prevents worrying about a particular
        # source code version when trying to reproduce exact parameters
        write_source_code(
            project_src,
            self.writer.subject_path + "/src/",
        )

    # code to run before human and computer trials
    def setup_trial(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        specified_trial=None,
        multithreaded=False,
    ):
        """
        Set the env class variables and select a trial (specified if provided, otherwise a random trial from the scenario name).

        This method should be called before running human and computer trials.
        Returns the trial selected (string).

        :param scenario_name: name of scenario (e.g. those defined in settings_trial.PARAMS)
        :param action_limit: number of actions permitted
        :param attempt_limit: number of attempts permitted
        :param specified_trial: optional specified trial. If none, get_trial is used to select trial
        :param multithreaded:
        :return: the selected_trial as returned by get_trial or select_trial
        """
        trial_selected = self.env.setup_trial(
            scenario_name,
            action_limit=action_limit,
            attempt_limit=attempt_limit,
            specified_trial=specified_trial,
            multiproc=multithreaded,
        )
        # mark agent as not finished, we may be continuing a already-written agent object
        self.finished = False

        return trial_selected

    def finish_trial(self, trial_selected, test_trial):
        """
        Finish trial and log it.
        Reset variables after finishing trial and call agent.finish_trial(). Add finished trial to completed_trials.

        :param trial_selected: trial to add to completed_trials
        :param test_trial: true if test trial
        :return:
        """
        # copy trial results from OpenLock env
        self.logger.cur_trial = self.env.cur_trial
        self.logger.finish_trial()
        self.write_trial(test_trial)
        self.env.finish_trial(trial_selected)

    def finish_attempt(self):
        self.env.finish_attempt()

    def finish_subject(self, strategy, transfer_strategy, agent=None):
        """
        Finish subject at current time, set strategy and transfer_strategy, call write_results().

        :param strategy:
        :param transfer_strategy:
        :return: Nothing
        """
        if agent is None:
            agent = self

        if agent.finished:
            print("Already finished agent, cannot finish again...")
            return

        agent.logger.finish(time.time())
        agent.logger.strategy = strategy
        agent.logger.transfer_strategy = transfer_strategy

        agent.write_agent(agent)

        agent.finished = True

    # todo: refactor/consolidate this with opt['trial_success'] returned by env.step()
    def determine_trial_finished(self, attempt_limit=None):
        if attempt_limit is None:
            attempt_limit = self.env.attempt_limit
        if self.human:
            if self.env.attempt_count >= attempt_limit or self.env.get_trial_success():
                return True
            else:
                return False
        else:
            # end if attempt limit reached
            if self.env.attempt_count >= attempt_limit:
                return True
            # trial is success and not forcing agent to use all attempts
            elif (
                self.params["full_attempt_limit"] is False
                and self.env.get_trial_success() is True
            ):
                return True
            return False

    def add_attempt(self):
        """
        Log the attempt

        :return: Nothing
        """
        self.logger.cur_trial.add_attempt()

    def get_current_attempt_logged_actions(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index("agent")
        actions = results[idx][agent_idx + 1 : len(results[idx])]
        action_labels = results[0][agent_idx + 1 : len(results[idx])]
        return actions, action_labels

    def get_current_attempt_logged_states(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index("agent")
        # frame is stored in 0
        states = results[idx][1:agent_idx]
        state_labels = results[0][1:agent_idx]
        return states, state_labels

    def get_last_attempt(self):
        trial, _ = self.get_last_trial()
        if trial.cur_attempt is not None:
            return trial.cur_attempt
        else:
            return trial.attempt_seq[-1]

    # only want results from a trial/
    def get_last_results(self):
        trial, _ = self.get_last_trial()
        if trial.cur_attempt is not None and trial.cur_attempt.results is not None:
            return trial.cur_attempt.results
        else:
            return trial.attempt_seq[-1].results

    def get_last_trial(self):
        if self.env.cur_trial is not None:
            return self.env.cur_trial, True
        else:
            return self.logger.trial_seq[-1], False

    def get_possible_trails(self, scenario_name=None):
        """
        Gets the possible list of trials based on the env's current scenario
        :return:
        """
        if scenario_name is None:
            assert (
                self.env is not None and self.env.scenario is not None
            ), "Scenario is none"
            scenario_name = self.env.scenario.name
        return get_possible_trials(scenario_name)

    def get_random_order_of_possible_trials(self, scenario_name=None):
        possible_trials = self.get_possible_trails(scenario_name)
        random.shuffle(possible_trials)
        return possible_trials

    def pretty_print_last_results(self):
        """
        Print results in an ASCII table.

        :return: nothing
        """
        results = self.get_last_results()
        table = texttable.Texttable()
        col_labels = results[0]
        table.set_cols_align(["l" for i in range(len(col_labels))])
        content = [col_labels]
        content.extend(results[1 : len(results)])
        table.add_rows(content)
        table.set_cols_width([12 for i in range(len(col_labels))])
        print(table.draw())

    @staticmethod
    def pre_instantiation_setup(params, bypass_confirmation=False):
        print("PARAMETERS:")
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(params)

        if not bypass_confirmation:
            input("Press Enter to confirm...")

        # copy the source code to the tmp directory - we need to save a copy when we start, not when we create an agent. Source code could change in between time program is launched and agents are created (especially if running multiple agents)

        # allows disabling writing the source code (used for replay)
        if params["src_dir"] is not None:
            write_source_code(ROOT_DIR, params["src_dir"])

        env = Agent.make_env(params)
        return env

    @staticmethod
    def make_env(params):
        # setup initial env
        env = gym.make("openlock-v1")
        env.use_physics = params["use_physics"]
        env.initialize_for_scenario(params["train_scenario_name"])
        if "effect_probabilities" in params.keys():
            env.set_effect_probabilities(params["effect_probabilities"])
        return env

    def write_agent(self, agent):
        """
        Log current agent state.

        :return: Nothing
        """
        self.writer.write(self.logger, agent)

    def write_trial(self, test_trial=False):
        """
        Log trial.

        :param test_trial: true if test trial, default: False
        :return: Nothing
        """
        self.writer.write_trial(self.logger, test_trial)

    def print_update(
        self,
        iter_num,
        trial_num,
        scenario_name,
        episode,
        episode_max,
        a_reward,
        t_reward,
        epsilon,
    ):
        """
        Print ID, iteration number, trial number, scenario, episode, attempt_reward, trial_reward, epsilon.

        :param iter_num:
        :param trial_num:
        :param scenario_name:
        :param episode:
        :param episode_max:
        :param a_reward:
        :param t_reward:
        :param epsilon:
        :return: Nothing
        """
        print(
            "ID: {}, iter {}, trial {}, scenario {}, episode: {}/{}, attempt_reward {}, trial_reward {}, e: {:.2}".format(
                self.subject_id,
                iter_num,
                trial_num,
                scenario_name,
                episode,
                episode_max,
                a_reward,
                t_reward,
                epsilon,
            )
        )

    def verify_fsm_matches_simulator(self, obs_space):
        """
        Ensure that the simulator data matches the FSM.

        :param obs_space:
        :return: obs_space
        """
        if obs_space is None:
            obs_space = ObservationSpace(len(self.env.world_def.get_levers()))
        sim_state, sim_labels = obs_space.create_discrete_observation_from_simulator(self.env)
        fsm_state, fsm_labels = obs_space.create_discrete_observation_from_fsm(self.env)
        try:
            assert sim_state == fsm_state
            assert sim_labels == fsm_labels
        except AssertionError:
            print("FSM does not match simulator data")
            print(sim_state)
            print(fsm_state)
            print(sim_labels)
            print(fsm_labels)
        return obs_space

    def plot_reward(self, reward, epoch):
        self.plot_value("attempt_reward", reward, epoch)

    def plot_value(self, name, xvalue, yvalue):
        self.writer.tensorboard_writer.write_scalar(name, yvalue, xvalue)

    @staticmethod
    def plot_rewards(rewards, epsilons, filename, width=12, height=6):
        plt.clf()
        assert len(epsilons) == len(rewards)
        moving_avg = Agent.compute_moving_average(rewards, 100)
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(width, height)
        plt.xlim((0, len(rewards)))
        r, = plt.plot(
            rewards,
            color="red",
            linestyle="-",
            linewidth=0.5,
            label="reward",
            alpha=0.5,
        )
        ave_r, = plt.plot(
            moving_avg, color="blue", linestyle="-", linewidth=0.8, label="avg_reward"
        )
        # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
        plt.legend([r, ave_r], ["reward", "average reward"])
        plt.ylabel("Reward")
        plt.xlabel("Episode #")
        plt.show()
        plt.savefig(filename)

    @staticmethod
    def show_rewards(rewards, epsilons, fig, width=12, height=6, window_size=1000):
        # sanity checks for plotting
        assert fig is not None
        assert len(epsilons) == len(rewards)
        if len(rewards) == 0:
            return

        plt.figure(fig.number)
        plt.clf()
        moving_avg = Agent.compute_moving_average(rewards, window_size)
        gcf = plt.gcf()
        ax = plt.gca()
        gcf.set_size_inches(width, height)
        plt.xlim((0, len(rewards)))
        r, = plt.plot(
            rewards,
            color="red",
            linestyle="-",
            linewidth=0.5,
            label="reward",
            alpha=0.5,
        )
        ave_r, = plt.plot(
            moving_avg, color="blue", linestyle="-", linewidth=0.8, label="avg_reward"
        )
        # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
        plt.legend([r, ave_r], ["reward", "average reward"])
        plt.ylabel("Reward")
        plt.xlabel("Episode #")
        plt.draw()
        plt.pause(0.1)

    def log_values(self, values_list, fig, taglist, title):
        """ log several lists of values with specific tags,
            values with different tag will be drawn in different figure

        Parameters
        ----------
        values_list : :obj:`list` of float
            list of list of value
        fig : :obj:`matplotlib.figure.Figure`
            fig for plotting values
        taglist : :obj:`list` of :obj:`str`
            list of tag for values
        title: : :obj:`str`
            title string

        Returns
        -------
        :obj:`matplotlib.figure.Figure`
        """

        def smooth(y, box_pts):
            box_pts = max(box_pts, 1) if len(y) > box_pts else 1
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode="valid")
            return y_smooth

        # sanity check
        assert fig is not None
        assert len(taglist) == len(values_list)

        fig.clf()
        num_plots = len(taglist)
        clrs = sns.color_palette("husl", 1)
        base = fig.add_subplot(111)
        base.spines["top"].set_color("none")
        base.spines["bottom"].set_color("none")
        base.spines["left"].set_color("none")
        base.spines["right"].set_color("none")
        base.tick_params(
            labelcolor="w", top="off", bottom="off", left="off", right="off"
        )
        with sns.axes_style("darkgrid"):
            axes = fig.subplots(1, num_plots)
            for ind, (tag, values) in enumerate(zip(taglist, values_list)):
                axes[ind].plot(
                    np.arange(len(values)), values, "-", c=clrs[0], alpha=0.3
                )
                # axes[ind].plot(np.arange(len(values)), smooth(values, len(values)//30), '-', c=clrs[0])
                res = smooth(values, 30)
                axes[ind].plot(np.arange(len(res)), res, "-", c=clrs[0])
                # axes[ind].plot(np.arange(len(values)), smooth(values, 30), '-', c=clrs[0])
                # axes[ind].fill_between(epochs, mins, maxs, alpha=0.3, facecolor=clrs[0])
                axes[ind].set_ylabel(tag)
        base.set_title(title)
        base.set_xlabel("Iteration")
        # plt.pause(0.001)
        fig.tight_layout()
        return fig, (taglist, values_list)

    @staticmethod
    def plot_rewards_trial_switch_points(
        rewards,
        epsilons,
        trial_switch_points,
        filename,
        plot_xticks=False,
        window_size=1000,
    ):
        plt.clf()
        assert len(epsilons) == len(rewards)
        moving_avg = Agent.compute_moving_average(rewards, window_size)
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(12, 6)
        plt.xlim((0, len(rewards)))
        # mark where the trials changed
        for trial_switch_point in trial_switch_points:
            plt.axvline(
                trial_switch_point,
                color="black",
                linewidth=0.5,
                linestyle="--",
                alpha=0.3,
            )
        r, = plt.plot(
            rewards,
            color="red",
            linestyle="-",
            linewidth=0.5,
            label="reward",
            alpha=0.5,
        )
        ave_r, = plt.plot(
            moving_avg, color="blue", linestyle="-", linewidth=0.8, label="avg_reward"
        )
        # e, = plt.plot(epsilons, color='blue', linestyle='--', alpha=0.5, label='epsilon')
        plt.legend([r, ave_r], ["reward", "average reward"])
        if plot_xticks:
            xtick_points, xtick_labels = Agent.create_xtick_labels(trial_switch_points)
            plt.xticks(xtick_points, xtick_labels)
            # vertical alignment of xtick labels
            va = [0 if x % 2 == 0 else -0.03 for x in range(len(xtick_points))]
            for t, y in zip(ax.get_xticklabels(), va):
                t.set_y(y)
        plt.ylabel("Reward")
        plt.xlabel("Episode # and trial #")
        plt.savefig(filename)

    @staticmethod
    def compute_moving_average(rewards, window):
        cur_window_size = 1
        moving_average = []
        for i in range(len(rewards) - 1):
            lower_idx = max(0, i - cur_window_size)
            average = sum(rewards[lower_idx : i + 1]) / cur_window_size
            moving_average.append(average)
            cur_window_size += 1
            if cur_window_size > window:
                cur_window_size = window
        return moving_average

    @staticmethod
    def create_xtick_labels(trial_switch_points):
        xtick_points = [0]
        xtick_labels = ["0"]
        prev_switch_point = 0
        trial_count = 1
        for trial_switch_point in trial_switch_points:
            xtick_point = (
                (trial_switch_point - prev_switch_point) / 2
            ) + prev_switch_point
            xtick_points.append(xtick_point)
            xtick_labels.append("trial " + str(trial_count))
            xtick_points.append(trial_switch_point)
            xtick_labels.append(str(trial_switch_point))
            trial_count += 1
            prev_switch_point = trial_switch_point
        return xtick_points, xtick_labels
