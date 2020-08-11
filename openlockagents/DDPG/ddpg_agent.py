import numpy as np
import os
import copy
import tensorflow.contrib.slim as slim
import tensorflow as tf

from openlockagents.common.agent import Agent, DEBUGGING


class DAgent(Agent):
    def __init__(self, name, env, state_size, action_size, params):
        super(DAgent, self).__init__(name, params, env)
        super(DAgent, self).setup_subject(human=False)

        self.params = params

        self.state_size = state_size
        self.action_size = action_size
        self.memory = np.zeros(
            (200000, self.state_size * 2 + self.action_size + 1), dtype=np.float32
        )
        self.pointer = 0
        self.gamma = params["gamma"]  # discount rate
        self.epsilon = params["epsilon"]  # exploration rate
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.learning_rate = params["learning_rate"]
        self.epsilons = []
        self.rewards = []
        self.trial_rewards = []
        self.trial_switch_points = []
        self.average_trial_rewards = []
        self.batch_size = params["batch_size"]
        self.train_num_iters = params["train_num_iters"]
        self.train_attempt_limit = params["train_attempt_limit"]
        self.train_action_limit = params["train_action_limit"]
        self.test_attempt_limit = params["test_attempt_limit"]
        self.test_action_limit = params["test_action_limit"]
        self.reward_mode = params["reward_mode"]
        self.last_batch_size = 0

    def finish_subject(
        self,
        strategy="Deep Q-Learning",
        transfer_strategy="Deep Q-Learning",
        agent=None,
    ):
        if agent is None:
            agent = self
        agent_cpy = copy.copy(agent)
        if hasattr(agent_cpy, "memory"):
            del agent_cpy.memory
        if hasattr(agent_cpy, "model"):
            del agent_cpy.model
        super(DAgent, self).finish_subject(strategy, transfer_strategy, agent_cpy)

    def remember(self, state, action, reward, next_state, done):
        self.memory.remember(state, action, reward, next_state)

    def act(self, state):
        action = self.sess.run(self.actor.a, feed_dict={self.actor.S: state})[
            0
        ]  # single action
        index = np.argmax(action)
        return index

    def save_reward(self, reward, trial_reward):
        self.epsilons.append(self.epsilon)
        self.rewards.append(reward)
        self.trial_rewards.append(trial_reward)

    # update the epsilon after every trial once it drops below epsilon_threshold
    def update_dynamic_epsilon(self, epsilon_threshold, new_epsilon, new_epsilon_decay):
        if self.epsilon < epsilon_threshold:
            self.epsilon = new_epsilon
            self.epsilon_decay = new_epsilon_decay

    def save_weights(self, save_dir, filename, sess):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save(sess, save_dir + "/" + filename)

    def save_agent(self, save_dir, testing, iter_num, trial_count, attempt_count, sess):
        if testing:
            save_str = (
                "/agent_test_i_"
                + str(iter_num)
                + "_t"
                + str(trial_count)
                + "_a"
                + str(attempt_count)
                + ".cptk"
            )
        else:
            save_str = (
                "/agent_i_"
                + str(iter_num)
                + "_t"
                + str(trial_count)
                + "_a"
                + str(attempt_count)
                + ".cptk"
            )
        self.save_weights(save_dir, save_str, sess)

    # load Keras weights (.h5)
    def load(self, name):
        self.saver.restore(name)

    # save Keras weights (.h5)
    def save(self, sess, name):
        self.saver.save(sess, name)

    # todo: this should probably be associated with the dqn/ddqn agents, but they both share this function
    # code to run a computer trial
    def run_trial_dqn(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        trial_count,
        iter_num,
        test_trial=False,
        specified_trial=None,
        fig=None,
        fig_update_rate=100,
    ):
        """
        Run a computer trial.

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param test_trial:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial
        )

        save_dir = self.writer.subject_path + "/models"

        print(
            "Scenario name: {}, iter num: {}, trial count: {}, trial name: {}".format(
                scenario_name, iter_num, trial_count, trial_selected
            )
        )

        trial_reward = 0
        attempt_count = 0
        attempt_reward = 0
        reward = 0
        while not self.determine_trial_finished(attempt_limit):
            done = False
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # run an attempt
            while not done:
                prev_attempt_reward = attempt_reward
                prev_reward = reward
                # self.env.render()

                action_idx = self.act(state)
                action_list = np.zeros(self.action_size)
                action_list[action_idx] = 1
                # convert idx to Action object (idx -> str -> Action)
                action = self.env.action_map[self.env.action_space[action_idx]]
                next_state, reward, done, opt = self.env.step(action)

                next_state = np.reshape(next_state, [1, self.state_size])

                self.remember(state, action_list, reward, next_state, done)
                # remember(state, action_idx, trial_reward, next_state, done)
                # self.env.render()

                trial_reward += reward
                attempt_reward += reward
                state = next_state

            self.finish_attempt()

            if DEBUGGING:
                self.print_update(
                    iter_num,
                    trial_count,
                    scenario_name,
                    self.env.attempt_count,
                    self.env.attempt_limit,
                    attempt_reward,
                    trial_reward,
                    self.epsilon,
                )
                print(self.env.cur_trial.attempt_seq[-1].action_seq)

            self.save_reward(attempt_reward, trial_reward)
            self.plot_reward(attempt_reward, self.total_attempt_count)

            assert (
                self.env.cur_trial.cur_attempt.cur_action is None
                and len(self.env.cur_trial.cur_attempt.action_seq) == 0
            )
            assert attempt_reward == self.env.cur_trial.attempt_seq[-1].reward

            attempt_reward = 0
            attempt_count += 1

            # replay to learn
            if len(self.memory) > self.batch_size:
                self.learn()

            # save s model
            # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
            if (
                self.env.attempt_count == 0
                or self.env.attempt_count == self.env.attempt_limit
            ):
                self.save_agent(
                    save_dir, test_trial, iter_num, trial_count, self.env.attempt_count
                )

        self.env.cur_trial.trial_reward = trial_reward
        self.finish_trial(trial_selected, test_trial=test_trial)

        self.ddpg_trial_sanity_checks()

        self.trial_switch_points.append(len(self.rewards))
        self.average_trial_rewards.append(trial_reward / self.env.attempt_count)

    def ddpg_trial_sanity_checks(self):
        """
        Used by dqn & ddqn trials to make sure attempt_seq and reward_seq are valid.

        :return: Nothing
        """
        last_trial, _ = self.get_last_trial()
        if len(self.trial_switch_points) > 0:
            assert (
                len(last_trial.attempt_seq)
                == len(self.rewards) - self.trial_switch_points[-1]
            )
            reward_agent = self.rewards[self.trial_switch_points[-1] :]
        else:
            assert len(last_trial.attempt_seq) == len(self.rewards)
            reward_agent = self.rewards[:]
        reward_seq = []
        for attempt in last_trial.attempt_seq:
            reward_seq.append(attempt.reward)
        assert reward_seq == reward_agent


class DDPGAgent(DAgent):
    def __init__(
        self, env, state_size, action_size, params, sess, name, capacity=200000
    ):
        super(DDPGAgent, self).__init__("DDPG", env, state_size, action_size, params)
        self.params = params
        self.memory = np.zeros(
            (200000, self.state_size + self.action_size + self.state_size + 1),
            dtype=np.float32,
        )
        self.sess = sess
        self.update_rate = 0.1
        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self._build_model(name)
        self.saver = tf.train.Saver()

        self.count = 0

    def finish_subject(self, strategy="DDPG", transfer_strategy="DDPG", agent=None):
        if agent is None:
            agent = self
        super(DAgent, self).finish_subject(strategy, transfer_strategy, agent)

    def act(self, state):
        weights = self.sess.run(self.a, feed_dict={self.S: state})[0]  # single action
        weights += np.random.uniform(-1.0, 1.0, self.action_size) * 0.005
        t = np.cumsum(weights)
        s = np.sum(weights)
        index = np.searchsorted(t, np.random.rand(1) * s)

        return index

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = slim.fully_connected(
                s,
                128,
                activation_fn=tf.nn.relu,
                biases_initializer=None,
                scope="l1",
                trainable=trainable,
            )
            net = slim.fully_connected(
                net,
                128,
                activation_fn=tf.nn.relu,
                biases_initializer=None,
                scope="l2",
                trainable=trainable,
            )
            a = slim.fully_connected(
                net,
                self.action_size,
                activation_fn=tf.nn.softmax,
                biases_initializer=None,
                scope="a",
                trainable=trainable,
            )
            return a

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            inputs = tf.concat((s, a), axis=1)
            net = slim.fully_connected(
                inputs,
                128,
                activation_fn=tf.nn.relu,
                biases_initializer=None,
                scope="w1",
                trainable=trainable,
            )

            net = slim.fully_connected(
                net,
                128,
                activation_fn=tf.nn.relu,
                biases_initializer=None,
                scope="w2",
                trainable=trainable,
            )
            net = slim.fully_connected(
                net,
                1,
                activation_fn=None,
                biases_initializer=None,
                scope="w3",
                trainable=trainable,
            )
            return net

    def _build_model(self, name):
        with tf.variable_scope(name):
            self.S = tf.placeholder(tf.float32, [None, self.state_size], "s")
            self.S_ = tf.placeholder(tf.float32, [None, self.state_size], "s_")
            self.R = tf.placeholder(tf.float32, [None, 1], "r")

            with tf.variable_scope("Actor"):
                self.a = self._build_actor(self.S, scope="eval", trainable=True)
                a_ = self._build_actor(self.S_, scope="target", trainable=False)
            with tf.variable_scope("Critic"):
                q = self._build_critic(self.S, self.a, scope="eval", trainable=True)
                q_ = self._build_critic(self.S_, a_, scope="target", trainable=False)

            if name != "init":
                self.td_error = tf.reduce_mean(
                    tf.squared_difference(self.R + self.gamma * q_, q)
                )
                # networks parameters
                self.params_a = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "/Actor/eval"
                )
                self.params_a_target = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "/Actor/target"
                )
                self.params_c = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "/Critic/eval"
                )
                self.params_c_target = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "/Critic/target"
                )

                self.update_target = [
                    self.update_target_graph(self.params_a, self.params_a_target),
                    self.update_target_graph(self.params_c, self.params_c_target),
                ]

                self.critic_train = self.trainer.minimize(
                    self.td_error, var_list=self.params_c
                )

                self.actor_loss = -tf.reduce_mean(q)
                self.actor_train = self.trainer.minimize(
                    self.actor_loss, var_list=self.params_a
                )

    def _remember(self, state, action, reward, next_state, done):
        data = np.hstack(
            (state, np.asarray([action]), np.asarray([[reward]]), next_state)
        )
        index = self.count % 200000
        self.memory[index, :] = data
        self.count += 1

    def learn(self):
        self.sess.run(self.update_target)
        indices = np.random.choice(200000, size=self.batch_size)
        data = self.memory[indices, :]
        bs = data[:, : self.state_size]
        ba = data[:, self.state_size : self.state_size + self.action_size]
        br = data[:, -self.state_size - 1 : -self.state_size]
        bs_ = data[:, -self.state_size :]

        self.sess.run(self.actor_train, {self.S: bs})
        self.sess.run(
            self.critic_train, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_}
        )

    def update_target_graph(self, from_vars, to_vars):
        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(
                tf.assign(
                    to_var,
                    (1 - self.update_rate) * to_var + self.update_rate * from_var,
                )
            )
        return op_holder

    # code to run a computer trial ddpg memory replay
    def run_trial_ddpg(
        self,
        scenario_name,
        action_limit,
        attempt_limit,
        trial_count,
        iter_num,
        test_trial=False,
        specified_trial=None,
        fig=None,
        fig_update_rate=100,
    ):
        """
        Run a computer trial (ddqg).

        :param scenario_name:
        :param action_limit:
        :param attempt_limit:
        :param trial_count:
        :param iter_num:
        :param test_trial:
        :param specified_trial:
        :param fig:
        :param fig_update_rate:
        :return:
        """
        self.env.human_agent = False
        trial_selected = self.setup_trial(
            scenario_name, action_limit, attempt_limit, specified_trial
        )

        save_dir = self.writer.subject_path + "/models"

        print(
            (
                "scenario_name: {}, trial_count: {}, trial_name: {}".format(
                    scenario_name, trial_count, trial_selected
                )
            )
        )

        trial_reward = 0
        attempt_count = 0
        attempt_reward = 0
        train_step = 0
        while not self.determine_trial_finished(attempt_limit):
            done = False
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            # run an attempt
            while not done:
                # self.env.render()

                action_idx = self.act(state)
                action_list = np.zeros(self.action_size, dtype=np.int64)
                action_list[action_idx] = 1
                action_idx = np.argmax(action_list)
                # convert idx to Action object (idx -> str -> Action)
                action = self.env.action_map[self.env.action_space[action_idx]]
                next_state, reward, done, opt = self.env.step(action)

                next_state = np.reshape(next_state, [1, self.state_size])

                self._remember(state, action_list, reward, next_state, done)
                # remember(state, action_idx, trial_reward, next_state, done)
                # self.env.render()

                trial_reward += reward
                attempt_reward += reward
                state = next_state

            self.finish_attempt()

            if DEBUGGING:
                self.print_update(
                    iter_num,
                    trial_count,
                    scenario_name,
                    self.env.attempt_count,
                    self.env.attempt_limit,
                    attempt_reward,
                    trial_reward,
                    self.epsilon,
                )
                print(self.env.cur_trial.attempt_seq[-1].action_seq)

            self.save_reward(attempt_reward, trial_reward)
            self.plot_reward(attempt_reward, self.total_attempt_count)

            assert attempt_reward == self.env.cur_trial.attempt_seq[-1].reward

            attempt_reward = 0
            attempt_count += 1

            # actually reset the env
            state = self.env.reset()

            # replay to learn
            if len(self.memory.data) > self.batch_size:
                self.learn()

            # save agent's model
            # if self.env.attempt_count % (self.env.attempt_limit/2) == 0 or self.env.attempt_count == self.env.attempt_limit or self.env.logger.cur_trial.success is True:
            if (
                self.env.attempt_count == 0
                or self.env.attempt_count == self.env.attempt_limit
            ):
                self.save_agent(
                    save_dir,
                    test_trial,
                    iter_num,
                    trial_count,
                    self.env.attempt_count,
                    self.sess,
                )

            train_step += 1
        indices = np.random.choice(200000, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, : self.state_size]
        ba = bt[:, self.state_size : self.state_size + self.action_size]
        br = bt[:, -self.state_size - 1 : -self.state_size]
        bs_ = bt[:, -self.state_size :]

        td_error, a_loss = self.sess.run(
            [self.td_error, self.actor_loss],
            {self.S: bs, self.a: ba, self.R: br, self.S_: bs_},
        )
        print(
            "TD_error:{td_error}, a_loss:{a_loss}".format(
                td_error=td_error, a_loss=a_loss
            )
        )
        self.env.cur_trial.trial_reward = trial_reward
        self.finish_trial(trial_selected, test_trial=test_trial)

        self.ddpg_trial_sanity_checks()

        self.trial_switch_points.append(len(self.rewards))
        self.average_trial_rewards.append(trial_reward / self.env.attempt_count)
