import torch
import os
import numpy as np

from openlockagents.common.agent import Agent, DEBUGGING

from openlockagents.TRPO.model import DiscretePolicy
from openlockagents.TRPO.model import Value
from openlockagents.TRPO.core import update_params
from openlockagents.TRPO.utils.replay_memory import Memory


class TRPOAgent(Agent):
    def __init__(self, env, state_size, action_size, params, require_log=True):
        """
        Init TRPO agent for OpenLock env

        :param env
        :param state_size
        :param action_size
        :param params
        """
        super(TRPOAgent, self).__init__("TRPO", params, env)
        if require_log:
            super(TRPOAgent, self).setup_subject(human=False)

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
        self.memory = Memory()
        self.gamma = params["gamma"]
        self.lr = params["learning_rate"]
        self.epsilon = params["epsilon"]
        self.l2_reg = params["l2_reg"]
        self.max_kl = params["max_kl"]
        self.damping = params["damping"]
        self.batch_size = params["batch_size"]
        self.use_gpu = params["use_gpu"] and torch.cuda.is_available()
        self.gpuid = params["gpuid"]
        self.tensor = torch.cuda.DoubleTensor if self.use_gpu else torch.DoubleTensor
        self.action_tensor = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor
        self.render = params["use_physics"]

        self.STEPS = 0

        self.env.lever_index_mode = "role"

        self._build_model()

    def _build_model(self):
        self.policy = DiscretePolicy(self.state_size, self.action_size)
        self.value = Value(self.state_size)
        if self.use_gpu:
            torch.cuda.set_device(self.gpuid)
            self.policy = self.policy.cuda()
            self.value = self.value.cuda()

    def run_trial_trpo(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        trial_count,
        iter_num,
        testing=False,
        specified_trial=None,
        fig=None,
        fig_update_rate=10,
    ):
        """
        Run a computer trial using TRPO.

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

    def update(self, batch):
        """
        Update the actor-critic model with TRPO
        Args:
        """
        update_params(
            self.policy,
            self.value,
            batch,
            self.tensor,
            self.action_tensor,
            self.gamma,
            self.epsilon,
            self.max_kl,
            self.damping,
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
        return int(self.policy.select_action(s, is_train=train)[0].cpu().numpy())

    def save(self, path, iter_num):
        fd = "/".join(path.split("/")[:-1])
        os.makedirs(fd, exist_ok=True)
        torch.save(
            self.policy.state_dict(),
            os.path.join(path, "{:06d}.policy".format(iter_num)),
        )
        torch.save(
            self.value.state_dict(), os.path.join(path, "{:06d}.value".format(iter_num))
        )

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + ".policy"))
        self.value.load_state_dict(torch.load(path + ".value"))

    def finish_subject(self, strategy="TRPO", transfer_strategy="TRPO"):
        super(TRPOAgent, self).finish_subject(strategy, transfer_strategy, self)
