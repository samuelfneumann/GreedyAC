#!/usr/bin/env python3

# Import modules
import time
from datetime import datetime
from copy import deepcopy
import numpy as np


class Experiment:
    """
    Class Experiment will run a single experiment, which consists of a single
    run of agent-environment interaction.
    """
    def __init__(self, agent, env, eval_env, eval_episodes,
                 total_timesteps, eval_interval_timesteps, max_episodes=-1):
        """
        Constructor

        Parameters
        ----------
        agent : baseAgent.BaseAgent
            The agent to run the experiment on
        env : environment.Environment
            The environment to use for the experiment
        eval_episodes : int
            The number of evaluation episodes to run when measuring offline
            performance
        total_timesteps : int
            The maximum number of allowable timesteps per experiment
        eval_interval_timesteps: int
            The interval of timesteps at which an agent's performance will be
            evaluated
        state_bins : tuple of int
            For the sequence of states used in each update, the number of bins
            per dimension with which to bin the states.
        min_state_values : array_like
             The minimum value of states along each dimension, used to encode
             states used in updates to count the number of times states are
             used in each update.
        max_state_values : array_like
             The maximum value of states along each dimension, used to encode
             states used in updates to count the number of times states are
             used in each update.
        action_bins : tuple of int
            For the sequence of actions used in each update, the number of bins
            per dimension with which to bin the actions.
        min_action_values : array_like
             The minimum value of actions along each dimension, used to encode
             actions used in updates to count the number of times actions are
             used in each update.
        max_state_values : array_like
             The maximum value of actions along each dimension, used to encode
             actions used in updates to count the number of times actions are
             used in each update.
        count_interval : int
            The interval of timesteps at which we will store the counts of
            state or action bins seen during training or used in updates. At
            each timestep, we determine which state/action bins were used in
            an update or seen at the current timestep. These values are
            accumulated so that the total number of times each bin was
            seen/used is stored up to the current timestep. This parameter
            controls the timestep interval at which these accumulated values
            should be checkpointed.
        max_episodes : int
            The maximum number of episodes to run. If <= 0, then there is no
            episode limit.
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.eval_env.monitor = False

        self.eval_episodes = eval_episodes
        self.max_episodes = max_episodes

        # Track the number of time steps
        self.timesteps_since_last_eval = 0
        self.eval_interval_timesteps = eval_interval_timesteps
        self.timesteps_elapsed = 0
        self.total_timesteps = total_timesteps

        # Keep track of number of training episodes
        self.train_episodes = 0

        # Track the returns seen at each training episode
        self.train_ep_return = []

        # Track the steps per each training episode
        self.train_ep_steps = []

        # Track the steps at which evaluation occurs
        self.timesteps_at_eval = []

        # Track the returns seen at each eval episode
        self.eval_ep_return = []

        # Track the number of evaluation steps taken in each evaluation episode
        self.eval_ep_steps = []

        # Anything the experiment tracks
        self.info = {}

        # Track the total training and evaluation time
        self.train_time = 0.0
        self.eval_time = 0.0

    def run(self):
        """
        Runs the experiment
        """
        # Count total run time
        start_run = time.time()
        print(f"Starting experiment at: {datetime.now()}")

        # Evaluate once at the beginning
        self.eval_time += self.eval()
        self.timesteps_at_eval.append(self.timesteps_elapsed)

        # Train
        i = 0
        while self.timesteps_elapsed < self.total_timesteps and \
                (self.train_episodes < self.max_episodes if
                 self.max_episodes > 0 else True):

            # Run the training episode and save the relevant info
            ep_reward, ep_steps, train_time = self.run_episode_train()
            self.train_ep_return.append(ep_reward)
            self.train_ep_steps.append(ep_steps)
            self.train_time += train_time
            print(f"=== Train ep: {i}, r: {ep_reward}, n_steps: {ep_steps}, " +
                  f"elapsed: {train_time}")
            i += 1

        # Evaluate once at the end
        self.eval_time += self.eval()
        self.timesteps_at_eval.append(self.timesteps_elapsed)

        end_run = time.time()
        print(f"End run at time {datetime.now()}")
        print(f"Total time taken: {end_run - start_run}")
        print(f"Training time: {self.train_time}")
        print(f"Evaluation time: {self.eval_time}")

        self.info["eval_episode_rewards"] = np.array(self.eval_ep_return)
        self.info["eval_episode_steps"] = np.array(self.eval_ep_steps)
        self.info["timesteps_at_eval"] = np.array(self.timesteps_at_eval)
        self.info["train_episode_steps"] = np.array(self.train_ep_steps)
        self.info["train_episode_rewards"] = np.array(self.train_ep_return)
        self.info["train_time"] = self.train_time
        self.info["eval_time"] = self.eval_time
        self.info["total_train_episodes"] = self.train_episodes

    def run_episode_train(self):
        """
        Runs a single training episode, saving the evaluation metrics in
        the corresponding instance variables.

        Returns
        -------
        float, int, float
            The return for the episode, the number of steps in the episode,
            and the total amount of training time for the episode
        """
        # Reset the agent
        self.agent.reset()

        self.train_episodes += 1

        # Track the sequences of states, rewards, and actions during training
        # episode_states = []
        episode_rewards = []
        # episode_actions = []

        start = time.time()
        episode_return = 0.0
        episode_steps = 0

        state, _ = self.env.reset()

        done = False
        action = self.agent.sample_action(state)

        while not done:
            # Evaluate offline at the appropriate intervals
            if self.timesteps_since_last_eval >= \
                    self.eval_interval_timesteps:
                self.eval_time += self.eval()
                self.timesteps_at_eval.append(self.timesteps_elapsed)

            # Sample the next transition
            next_state, reward, done, info = self.env.step(action)
            episode_steps += 1

            # episode_states.append(next_state_info["orig_state"])
            episode_rewards.append(reward)
            episode_return += reward

            # Compute the done mask, which is 1 if the episode terminated
            # without the goal being reached or the episode is incomplete,
            # and 0 if the agent reached the goal or terminal state
            if self.env.steps_per_episode <= 1:
                done_mask = 0
            else:
                if episode_steps <= self.env.steps_per_episode and done and \
                        not info["steps_exceeded"]:
                    done_mask = 0
                else:
                    done_mask = 1

            # Update agent
            self.agent.update(state, action, reward, next_state, done_mask)

            # Continue the episode if not done
            if not done:
                action = self.agent.sample_action(next_state)
            state = next_state

            # Keep track of the timesteps since we last evaluated so we know
            # when to evaluate again
            self.timesteps_since_last_eval += 1

            # Keep track of timesteps since we train for a specified number of
            # timesteps
            self.timesteps_elapsed += 1

            # Stop if we are at the max allowable timesteps
            if self.timesteps_elapsed >= self.total_timesteps:
                break

        end = time.time()

        return episode_return, episode_steps, (end-start)

    def eval(self):
        """
        Evaluates the agent's performance offline, for the appropriate number
        of offline episodes as determined by the self.eval_episodes
        instance variable.

        Returns
        -------
        float
            The total amount of evaluation time
        """
        self.timesteps_since_last_eval = 0

        # Set the agent to evaluation mode
        self.agent.eval()

        # Save the episodic return and the number of steps per episode
        temp_rewards_per_episode = []
        episode_steps = []
        eval_session_time = 0.0

        # Evaluate offline
        for i in range(self.eval_episodes):
            eval_start_time = time.time()
            episode_reward, num_steps = self.run_episode_eval()
            eval_end_time = time.time()

            # Save the evaluation data
            temp_rewards_per_episode.append(episode_reward)
            episode_steps.append(num_steps)

            # Calculate time
            eval_elapsed_time = eval_end_time - eval_start_time
            eval_session_time += eval_elapsed_time

            # Display the offline episodic return
            print("=== EVAL ep: " + str(i) + ", r: " +
                  str(episode_reward) + ", n_steps: " + str(num_steps) +
                  ", elapsed: " +
                  time.strftime("%H:%M:%S", time.gmtime(eval_elapsed_time)))

        # Save evaluation data
        self.eval_ep_return.append(temp_rewards_per_episode)
        self.eval_ep_steps.append(episode_steps)

        self.eval_time += eval_session_time

        # Return the agent to training mode
        self.agent.train()

        return eval_session_time

    def run_episode_eval(self):
        """
        Runs a single evaluation episode.

        Returns
        -------
        float, int, list
            The episodic return and number of steps and the sequence of states,
            rewards, and actions during the episode
        """
        state, _ = self.eval_env.reset()

        episode_return = 0.0
        episode_steps = 0
        done = False

        action = self.agent.sample_action(state)

        while not done:
            next_state, reward, done, _ = self.eval_env.step(action)

            episode_return += reward

            if not done:
                action = self.agent.sample_action(next_state)

            state = next_state
            episode_steps += 1

        return episode_return, episode_steps
