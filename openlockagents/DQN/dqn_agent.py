import torch
import os
import numpy as np

from openlockagents.common.agent import Agent, DEBUGGING

from openlockagents.DQN.model import QDiscretePolicy
from openlockagents.DQN.model import Value
from openlockagents.DQN.core import update_params
from openlockagents.DQN.utils.replay_memory import *


class DQNAgent(Agent):
    def __init__(self, env, state_size, action_size, params, require_log=True):
        """
        Init DQN agent for OpenLock env

        :param env
        :param state_size
        :param action_size
        :param params
        """
        super(DQNAgent, self).__init__("DQN", params, env)
        if require_log:
            super(DQNAgent, self).setup_subject(human=False)

        self.params = params
        self.state_size = state_size
        self.action_size = action_size
        self.attempt_rewards = []
        self.trial_rewards = []
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.trial_percent_solution_found = []
        self.trial_percent_attempt_success = []
        self.trial_length = []
        self.max_mem_size = params["max_mem_size"]
        self.gamma = params["gamma"]
        self.lr = params["learning_rate"]
        self.l2_reg = params["l2_reg"]
        self.batch_size = params["batch_size"]
        self.eps_start = params["eps_start"]
        self.eps_end = params["eps_end"]
        self.eps_decay = params["eps_decay"]
        self.use_gpu = params["use_gpu"] and torch.cuda.is_available()
        self.gpuid = params["gpuid"]
        self.tensor = torch.cuda.DoubleTensor if self.use_gpu else torch.DoubleTensor
        self.action_tensor = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor
        self.render = params["use_physics"]
        self.prioritized_replay = params["prioritized_replay"]
        if self.prioritized_replay:
            self.memory = PrioritizedMemory(limit=self.max_mem_size)
        else:
            self.memory = Memory(limit=self.max_mem_size)

        self.STEPS = 0

        self.env.lever_index_mode = "role"

        self._build_model()

    def _build_model(self):
        self.q_net = Value(self.state_size, self.action_size)
        self.policy = QDiscretePolicy(
            self.q_net, self.eps_start, self.eps_end, self.eps_decay
        )
        self.target_q_net = Value(self.state_size, self.action_size)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        if self.use_gpu:
            torch.cuda.set_device(self.gpuid)
            self.policy = self.policy.cuda()
            self.target_q_net = self.target_q_net.cuda()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def run_trial_dqn(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        trial_count,
        iter_num,
        testing=False,
        specified_trial=None,
        fig=None,
        fig_update_rate=100,
    ):
        """
        Run a computer trial using DQN.

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param testing:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial
        )

        # print('Scenario name: {}, iter num: {}, trial count: {}, trial name: {}'.format(scenario_name, iter_num, trial_count, trial_selected))

        trial_reward = 0
        self.env.attempt_count = 0
        attempt_reward = 0
        reward = 0
        attempt_success_count = 0
        while not self.determine_trial_finished(attempt_limit):
            done = False
            state = self.env.reset()
            while not done:
                prev_attempt_reward = attempt_reward
                prev_reward = reward

                action_idx = self.act(state, train=True)
                # convert idx to Action object (idx -> str -> Action)
                action = self.env.action_map[self.env.action_space[action_idx]]
                next_state, reward, done, opt = self.env.step(action)

                mask = 0 if done else 1
                self.memory.push(state, action_idx, mask, next_state, reward)

                if self.render:
                    self.env.render()
                trial_reward += reward
                attempt_reward += reward
                state = next_state

            self.finish_attempt()

            if DEBUGGING:
                pass
                # self.print_update(iter_num, trial_count, scenario_name, self.env.attempt_count, self.env.attempt_limit, attempt_reward, trial_reward, 1.0)
                # print(self.logger.cur_trial.attempt_seq[-1].action_seq)

            assert (
                self.env.cur_trial.cur_attempt.cur_action is None
                and len(self.env.cur_trial.cur_attempt.action_seq) == 0
            )
            assert attempt_reward == self.env.cur_trial.attempt_seq[-1].reward

            self.attempt_rewards.append(attempt_reward)

            attempt_reward = 0
            self.total_attempt_count += 1

            if opt["attempt_success"]:
                attempt_success_count += 1

            # if fig is not None and self.env.attempt_count % fig_update_rate == 0:
            #     fig = self.log_values([self.average_trial_rewards, self.attempt_rewards],
            #                           fig,
            #                           ['Average Trial Reward', 'Attempt Reward'])

        print(
            "Trial end, avg_reward:{}, solutions found:{}/{}".format(
                trial_reward / attempt_limit,
                len(self.env.get_completed_solutions()),
                len(self.env.get_solutions()),
            )
        )
        self.env.cur_trial.trial_reward = trial_reward
        self.trial_rewards.append(trial_reward)
        self.average_trial_rewards.append(trial_reward / self.env.attempt_count)
        self.trial_switch_points.append(len(self.attempt_rewards))
        self.trial_percent_solution_found.append(
            len(self.env.get_completed_solutions()) / len(self.env.get_solutions())
        )
        self.trial_percent_attempt_success.append(
            attempt_success_count / self.env.attempt_count
        )
        self.trial_length.append(self.env.attempt_count)

        self.finish_trial(trial_selected, test_trial=testing)

    def update(self, batch, i_iter):
        """
        Update the actor-critic model with DQN
        Args:
        """
        update_params(
            self.q_net,
            self.target_q_net,
            self.optimizer,
            batch,
            self.tensor,
            self.action_tensor,
            self.gamma,
            self.l2_reg,
        )

    def act(self, s, train=True):
        """
        Choose an action to take on state s
        Args:
            s: the state
            train: if train or test
        Ret:
            act: the chosen action to take
        """
        assert len(s.shape) == 1  # FIXME
        s = self.tensor(s.astype(np.float32))
        s = s.unsqueeze(0)
        # FIXME: we need to set self.policy.iter to make \epsilon-greedy work
        return int(self.policy.select_action(s, is_train=train)[0].cpu().numpy())

    def save(self, path, iter_num):
        fd = "/".join(path.split("/")[:-1])
        os.makedirs(fd, exist_ok=True)
        torch.save(
            self.policy.state_dict(),
            os.path.join(path, "{:06d}.policy".format(iter_num)),
        )

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".policy"))
        self.q_net = self.policy.q_net
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def finish_subject(self, strategy="DQN", transfer_strategy="DQN"):
        super(DQNAgent, self).finish_subject(strategy, transfer_strategy, self)
