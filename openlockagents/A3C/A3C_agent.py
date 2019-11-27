import numpy as np
import random
import os
import copy
from openlockagents.agent import Agent, DEBUGGING

from collections import deque
import tensorflow as tf
import scipy.signal
from matplotlib import pyplot as plt

from openlockagents.A3C.ac_network import AC_Network


NUM_DYNAMIC = 2


class ActorCriticAgent(Agent):
    def __init__(self, env, state_size, action_size, name, params):
        super(ActorCriticAgent, self).__init__("A3C", params, env)
        super(ActorCriticAgent, self).setup_subject(human=False)

        self.params = params

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = params["gamma"]  # discount rate
        self.epsilon = params["epsilon"]  # exploration rate
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.learning_rate = params["learning_rate"]
        self.epsilons = []
        self.rewards = []
        self.name = name
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
        self.epsilon_dynamic = 0

    # code for a3c
    def run_trial_a3c(
        self,
        sess,
        global_episodes,
        number,
        testing_trial,
        params,
        coord,
        attempt_limit,
        scenario_name,
        trial_count,
        is_test,
        a_size,
        MINI_BATCH,
        gamma,
        episode_rewards,
        episode_lengths,
        episode_mean_values,
        summary_writer,
        name,
        saver,
        model_path,
        REWARD_FACTOR,
        fig=None,
    ):
        """
        Run a trial for a3c.

        :param sess:
        :param global_episodes:
        :param number:
        :param testing_trial:
        :param params:
        :param coord:
        :param attempt_limit:
        :param scenario_name:
        :param trial_count:
        :param is_test:
        :param a_size:
        :param MINI_BATCH:
        :param gamma:
        :param episode_rewards:
        :param episode_lengths:
        :param episode_mean_values:
        :param summary_writer:
        :param name:
        :param saver:
        :param model_path:
        :param REWARD_FACTOR:
        :param fig:
        :return:
        """
        self.env.human_agent = False
        episode_count = sess.run(global_episodes)
        increment = global_episodes.assign_add(1)
        total_steps = 0
        print("Starting worker " + str(number))
        with sess.as_default(), sess.graph.as_default():

            sess.run(self.update_target_graph("global", name))
            episode_buffer = []
            episode_mini_buffer = []
            episode_values = []
            episode_states = []
            episode_reward = 0
            attempt_reward = 0
            episode_step_count = 0

            if not testing_trial:
                trial_selected = self.setup_trial(
                    params["train_scenario_name"],
                    params["train_action_limit"],
                    params["train_attempt_limit"],
                    multithreaded=True,
                )

            else:
                trial_selected = self.setup_trial(
                    params["test_scenario_name"],
                    params["test_action_limit"],
                    params["test_attempt_limit"],
                    specified_trial="trial7",
                    multithreaded=True,
                )
            if name == "worker_0":
                print(
                    "scenario_name: {}, trial_count: {}, trial_name: {}".format(
                        scenario_name, trial_count, trial_selected
                    )
                )
            done = False
            state = self.env.reset()
            rnn_state = self.local_AC.state_init

            while not coord.should_stop():
                # end if attempt limit reached
                sess.run(self.update_target_graph("global", name))
                if self.env.attempt_count >= attempt_limit or (
                    self.params["full_attempt_limit"] is False
                    and self.logger.cur_trial.success is True
                ):

                    episode_buffer = []
                    episode_mini_buffer = []
                    episode_values = []
                    episode_states = []
                    episode_reward = 0
                    attempt_reward = 0
                    episode_step_count = 0

                    if not testing_trial:
                        trial_selected = self.setup_trial(
                            params["train_scenario_name"],
                            params["train_action_limit"],
                            params["train_attempt_limit"],
                            multithreaded=True,
                        )

                    else:
                        trial_selected = self.setup_trial(
                            params["test_scenario_name"],
                            params["test_action_limit"],
                            params["test_attempt_limit"],
                            specified_trial="trial7",
                            multithreaded=True,
                        )
                    if name == "worker_0":
                        print(
                            "scenario_name: {}, trial_count: {}, trial_name: {}".format(
                                scenario_name, trial_count, trial_selected
                            )
                        )
                    # todo: MJE: what is the purpose of setting done to False here?
                    done = False
                    state = self.env.reset()
                    rnn_state = self.local_AC.state_init

                # Run an episode
                while not self.determine_trial_finished(attempt_limit):
                    done = False
                    state = self.env.reset()
                    # run an attempt
                    while not done:
                        episode_states.append(state)
                        if is_test:
                            self.env.render()

                        # Get preferred action distribution
                        a_dist, v, rnn_state = sess.run(
                            [
                                self.local_AC.policy,
                                self.local_AC.value,
                                self.local_AC.state_out,
                            ],
                            feed_dict={
                                self.local_AC.inputs: [state],
                                self.local_AC.state_in[0]: rnn_state[0],
                                self.local_AC.state_in[1]: rnn_state[1],
                            },
                        )

                        a0 = self.weighted_pick(
                            a_dist[0], 1, self.epsilon
                        )  # Use stochastic distribution sampling
                        # if is_test:
                        #    a0 = np.argmax(a_dist[0])  # Use maximum when testing
                        a = np.zeros(a_size)
                        a[a0] = 1
                        action_idx = np.argmax(a)
                        action = self.env.action_map[self.env.action_space[action_idx]]

                        next_state, reward, done, opt = self.env.step(action)
                        episode_reward += reward
                        attempt_reward += reward

                        episode_buffer.append(
                            [state, a, reward, next_state, done, v[0, 0]]
                        )
                        episode_mini_buffer.append(
                            [state, a, reward, next_state, done, v[0, 0]]
                        )

                        episode_values.append(v[0, 0])

                        state = next_state
                        total_steps += 1
                        episode_step_count += 1

                    self.finish_attempt(multithread=True)

                    # episode is finished
                    if DEBUGGING:
                        self.env.cur_trial.attempt_seq[-1].reward = attempt_reward
                    self.save_reward(attempt_reward, episode_reward)
                    self.plot_reward(attempt_reward, self.total_attempt_count)
                    attempt_reward = 0

                    # Train on mini batches from episode
                    if len(episode_mini_buffer) == MINI_BATCH:
                        v1 = sess.run(
                            [self.local_AC.value],
                            feed_dict={
                                self.local_AC.inputs: [state],
                                self.local_AC.state_in[0]: rnn_state[0],
                                self.local_AC.state_in[1]: rnn_state[1],
                            },
                        )
                        v_l, p_l, e_l, g_n, v_n = self.train(
                            episode_mini_buffer, sess, gamma, v1[0][0], REWARD_FACTOR
                        )
                        episode_mini_buffer = []

                    env_reset, next_state = self.finish_action(
                        next_state, multithread=True
                    )

                    state = next_state
                    total_steps += 1
                    episode_step_count += 1

                    if env_reset:
                        self.logger.cur_trial.attempt_seq[-1].reward = attempt_reward
                        self.save_reward(attempt_reward, episode_reward)
                        self.plot_reward(attempt_reward, self.total_attempt_count)
                        attempt_reward = 0

                self.logger.cur_trial.trial_reward = attempt_limit
                self.finish_trial(trial_selected, test_trial=False)
                self.trial_switch_points.append(len(self.rewards))
                self.average_trial_rewards.append(
                    attempt_reward / self.env.attempt_count
                )

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step_count)

                if (
                    episode_count % 50 == 0
                    and not episode_count % 1000 == 0
                    and not is_test
                ):
                    mean_reward = np.mean(episode_rewards[-5:])
                    mean_length = np.mean(episode_lengths[-5:])
                    mean_value = np.mean(episode_mean_values[-5:])
                    summary = tf.Summary()

                    # summary.text.add(tag='Scenario name', simple_value=str(self.env.scenario.name))
                    # summary.text.add(tag='trial count', simple_value=str(trial_count))
                    # summary.text.add(tag='trial name', simple_value=str(trial_selected))
                    summary.value.add(
                        tag="Perf/Reward", simple_value=float(mean_reward)
                    )
                    summary.value.add(
                        tag="Perf/Length", simple_value=float(mean_length)
                    )
                    summary.value.add(tag="Perf/Value", simple_value=float(mean_value))
                    summary.value.add(tag="Losses/Value Loss", simple_value=float(v_l))
                    summary.value.add(tag="Losses/Policy Loss", simple_value=float(p_l))
                    summary.value.add(tag="Losses/Entropy", simple_value=float(e_l))
                    summary.value.add(tag="Losses/Grad Norm", simple_value=float(g_n))
                    summary.value.add(tag="Losses/Var Norm", simple_value=float(v_n))
                    summary_writer.add_summary(summary, episode_count)

                    summary_writer.flush()

                if name == "worker_0":
                    if episode_count % 20 == 0 and not is_test:
                        saver.save(
                            sess, model_path + "/model-" + str(episode_count) + ".cptk"
                        )
                    # creating the figure
                    if episode_count % 20 == 0:
                        saver.save(
                            sess, model_path + "/model-" + str(episode_count) + ".cptk"
                        )
                        if fig == None:
                            fig = plt.figure()
                            fig.set_size_inches(12, 6)
                        else:
                            self.show_rewards(self.rewards, self.epsilons, fig)
                    if (episode_count) % 1 == 0:
                        print(
                            "| Reward: " + str(episode_reward),
                            " | Episode",
                            episode_count,
                            " | Epsilon",
                            self.epsilon,
                        )
                    if (episode_count) % 50 == 0 and episode_count != 0:
                        self.plot_rewards(
                            self.rewards,
                            self.epsilons,
                            model_path
                            + "/"
                            + str(episode_count)
                            + "training_rewards.png",
                        )
                    sess.run(increment)  # Next global episode

                # todo: MJE: should this line be here?
                self.update_dynamic_epsilon(
                    self.epsilon_min,
                    params["dynamic_epsilon_max"],
                    params["dynamic_epsilon_decay"],
                )

                episode_count += 1
                trial_count += 1

    def update_dynamic_epsilon(self, epsilon_threshold, new_epsilon, new_epsilon_decay):
        if self.epsilon < epsilon_threshold and self.epsilon_dynamic < NUM_DYNAMIC:
            self.epsilon = new_epsilon
            self.epsilon_decay = new_epsilon_decay
            self.epsilon_dynamic += 1

    def finish_subject(self, strategy="A3C-agent", transfer_strategy="A3C-agent"):
        agent_cpy = copy.copy(self)
        if hasattr(agent_cpy, "memory"):
            del agent_cpy.memory
        if hasattr(agent_cpy, "model"):
            del agent_cpy.model
        if hasattr(agent_cpy, "target_model"):
            del agent_cpy.target_model
        super(ActorCriticAgent, self).finish_subject(strategy, transfer_strategy)

    def save_reward(self, reward, trial_reward):
        self.epsilons.append(self.epsilon)
        self.rewards.append(reward)
        self.trial_rewards.append(trial_reward)

    def save_weights(self, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save(save_dir + "/" + filename)

    def save_agent(self, save_dir, testing, iter_num, trial_count, attempt_count):
        if testing:
            save_str = (
                "/agent_test_i_"
                + str(iter_num)
                + "_t"
                + str(trial_count)
                + "_a"
                + str(attempt_count)
                + ".h5"
            )
        else:
            save_str = (
                "/agent_i_"
                + str(iter_num)
                + "_t"
                + str(trial_count)
                + "_a"
                + str(attempt_count)
                + ".h5"
            )
        self.save_weights(save_dir, save_str)


class A3CAgent(ActorCriticAgent):
    def __init__(self, env, state_size, action_size, name, params):
        super(A3CAgent, self).__init__(env, state_size, action_size, name, params)
        self.local_AC = self._build_model()
        self.name = name

    def finish_subject(self, strategy="A3C_LSTM", transfer_strategy="A3C_LSTM"):
        super(A3CAgent, self).finish_subject(strategy, transfer_strategy)

    def _build_model(self):
        cell_unit = 256
        trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return AC_Network(
            self.state_size, self.action_size, self.name, trainer, cell_unit
        )

    def train(self, rollout, sess, gamma, r, REWARD_FACTOR):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        rewards_list = np.asarray(rewards.tolist() + [r]) * REWARD_FACTOR
        discounted_rewards = self.discounting(rewards_list, gamma)[:-1]

        # Advantage estimation
        # JS, P Moritz, S Levine, M Jordan, P Abbeel,
        # "High-dimensional continuous control using generalized advantage estimation."
        # arXiv preprint arXiv:1506.02438 (2015).
        values_list = np.asarray(values.tolist() + [r]) * REWARD_FACTOR
        advantages = rewards + gamma * values_list[1:] - values_list[:-1]
        discounted_advantages = self.discounting(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        # sess.run(self.local_AC.reset_state_op)

        rnn_state = self.local_AC.state_init
        feed_dict = {
            self.local_AC.target_v: discounted_rewards,
            self.local_AC.inputs: np.vstack(states),
            self.local_AC.actions: np.vstack(actions),
            self.local_AC.advantages: discounted_advantages,
            self.local_AC.state_in[0]: rnn_state[0],
            self.local_AC.state_in[1]: rnn_state[1],
        }
        v_l, p_l, e_l, g_n, v_n, _ = sess.run(
            [
                self.local_AC.value_loss,
                self.local_AC.policy_loss,
                self.local_AC.entropy,
                self.local_AC.grad_norms,
                self.local_AC.var_norms,
                self.local_AC.apply_grads,
            ],
            feed_dict=feed_dict,
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    # Copies one set of variables to another.
    # Used to set worker network parameters to those of global network.
    def update_target_graph(self, from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    # Weighted random selection returns n_picks random indexes.
    # the chance to pick the index i is give by the weight weights[i].
    def weighted_pick(self, weights, n_picks, epsilon=0.005):
        if np.random.rand(1) > epsilon:

            t = np.cumsum(weights)
            s = np.sum(weights)
            index = np.searchsorted(t, np.random.rand(n_picks) * s)
        else:
            index = random.randrange(self.action_size)
        return index

    # Discounting function used to calculate discounted returns.
    def discounting(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    # Normalization of inputs and outputs
    def norm(self, x, upper, lower=0.0):
        return (x - lower) / max((upper - lower), 1e-12)
