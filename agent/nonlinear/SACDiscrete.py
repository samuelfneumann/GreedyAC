import os
from gym.spaces import Box
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.MLP import Softmax
from agent.nonlinear.value_function.MLP import DoubleQ, Q
from utils.experience_replay import TorchBuffer as ExperienceReplay


class SACDiscrete(BaseAgent):
    def __init__(self, env, gamma, tau, alpha, policy, target_update_interval,
                 critic_lr, actor_lr_scale, actor_hidden_dim,
                 critic_hidden_dim, replay_capacity, seed, batch_size, betas,
                 double_q=True, soft_q=True, cuda=False, clip_stddev=1000,
                 init=None, activation="relu"):
        """
        Constructor

        Parameters
        ----------
        env : gym.Environment
            The environment to run on
        gamma : float
            The discount factor
        tau : float
            The weight of the weighted average, which performs the soft update
            to the target critic network's parameters toward the critic
            network's parameters, that is: target_parameters =
            ((1 - τ) * target_parameters) + (τ * source_parameters)
        alpha : float
            The entropy regularization temperature. See equation (1) in paper.
        policy : str
            The type of policy, currently, only support "gaussian"
        target_update_interval : int
            The number of updates to perform before the target critic network
            is updated toward the critic network
        critic_lr : float
            The critic learning rate
        actor_lr : float
            The actor learning rate
        actor_hidden_dim : int
            The number of hidden units in the actor's neural network
        critic_hidden_dim : int
            The number of hidden units in the critic's neural network
        replay_capacity : int
            The number of transitions stored in the replay buffer
        seed : int
            The random seed so that random samples of batches are repeatable
        batch_size : int
            The number of elements in a batch for the batch update
        cuda : bool, optional
            Whether or not cuda should be used for training, by default False.
            Note that if True, cuda is only utilized if available.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        action_space = env.action_space
        obs_space = env.observation_space
        if isinstance(action_space, Box):
            raise ValueError("SACDiscrete can only be used with " +
                             "discrete actions")

        super().__init__()
        self.batch = True

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.double_q = double_q
        self.soft_q = soft_q
        self.num_actions = action_space.n

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        # Keep a replay buffer
        action_shape = 1
        obs_dim = obs_space.shape
        self.replay = ExperienceReplay(replay_capacity, seed, obs_dim,
                                       action_shape, self.device)
        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        num_inputs = obs_space.shape[0]
        self._init_critic(obs_space, critic_hidden_dim, init, activation,
                          critic_lr, betas)

        self.policy_type = policy.lower()
        self._init_policy(obs_space, action_space, actor_hidden_dim, init,
                          activation, actor_lr_scale * critic_lr, betas,
                          clip_stddev)

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action, _, _ = self.policy.sample(state)
            act = action.detach().cpu().numpy()[0]
            return int(act[0])
        else:
            _, log_prob, _ = self.policy.sample(state)
            return log_prob.argmax().item()

    def update(self, state, action, reward, next_state, done_mask):
        # Adjust action to ensure it can be sent to the experience replay
        # buffer properly
        action = np.array([action])

        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            # Not enough samples in buffer
            return

        self._update_critic(state_batch, action_batch, reward_batch,
                            next_state_batch, mask_batch)

        self._update_actor(state_batch, action_batch, reward_batch,
                           next_state_batch, mask_batch)

    def reset(self):
        pass

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        pass

    def load_model(self, actor_path, critic_path):
        pass

    def get_parameters(self):
        pass

    def _init_critic(self, obs_space, critic_hidden_dim, init,
                     activation, critic_lr, betas):
        """
        Initializes the critic
        """
        num_inputs = obs_space.shape[0]

        if self.double_q:
            critic_type = DoubleQ
        else:
            critic_type = Q

        self.critic = critic_type(num_inputs, 1, critic_hidden_dim, init,
                                  activation).to(device=self.device)

        self.critic_target = critic_type(num_inputs, 1, critic_hidden_dim,
                                         init, activation).to(self.device)

        # Ensure critic and target critic share the same parameters at the
        # beginning of training
        nn_utils.hard_update(self.critic_target, self.critic)

        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=critic_lr,
            betas=betas,
        )

    def _init_policy(self, obs_space, action_space, actor_hidden_dim, init,
                     activation, actor_lr, betas, clip_stddev):
        """
        Initializes the policy
        """
        num_inputs = obs_space.shape[0]
        num_actions = action_space.n

        if self.policy_type == "softmax":
            self.policy = Softmax(num_inputs, num_actions, actor_hidden_dim,
                                  activation, init).to(self.device)

        else:
            raise NotImplementedError(f"policy {self.policy_type} unknown")

        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                 betas=betas)

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        if self.double_q:
            self._update_double_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch,)
        else:
            self._update_single_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        # Increment the running total of updates and update the critic target
        # if needed
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

    def _update_double_critic(self, state_batch, action_batch,
                              reward_batch, next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a double Q
        critic.
        """
        if not self.double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = \
                    self.policy.sample(next_state_batch)

            next_q1, next_q2 = self.critic_target(
                next_state_batch,
                next_state_action,
            )

            next_q = torch.min(next_q1, next_q2)

            if self.soft_q:
                next_q -= self.alpha * next_state_log_pi

            q_target = reward_batch + mask_batch * self.gamma * next_q

        q1, q2 = self.critic(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

    def _update_single_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a single Q
        critic.
        """
        if self.double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = \
                    self.policy.sample(next_state_batch)

            next_q = self.critic_target(next_state_batch, next_state_action)

            if self.soft_q:
                next_q -= self.alpha * next_state_log_pi

            q_target = reward_batch + mask_batch * self.gamma * next_q

        q = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q, q_target)

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

    def _get_q(self, state_batch, action_batch):
        """
        Gets the Q values for `action_batch` actions in `state_batch` states
        from the critic, rather than the target critic.

        Parameters
        ----------
        state_batch : torch.Tensor
            The batch of states to calculate the action values in. Of the form
            (batch_size, state_dims).
        action_batch : torch.Tensor
            The batch of actions to calculate the action values of in each
            state. Of the form (batch_size, action_dims).
        """
        if self.double_q:
            q1, q2 = self.critic(state_batch, action_batch)
            return torch.min(q1, q2)
        else:
            return self.critic(state_batch, action_batch)

    def _update_actor(self, state_batch, action_batch, reward_batch,
                      next_state_batch, mask_batch):
        # Calculate the actor loss using Eqn(5) in FKL/RKL paper
        # Repeat the state for each action
        state_batch = state_batch.repeat_interleave(self.num_actions, dim=0)
        actions = torch.tensor([n for n in range(self.num_actions)])
        actions = actions.repeat(self.batch_size)
        actions = actions.unsqueeze(-1)

        with torch.no_grad():
            q = self._get_q(state_batch, actions)

        log_prob = self.policy.log_prob(state_batch, actions)
        prob = log_prob.exp()

        with torch.no_grad():
            scale = q - log_prob * self.alpha
        policy_loss = prob * scale
        policy_loss = policy_loss.reshape([self.batch_size, self.num_actions])
        policy_loss = -policy_loss.sum(dim=1).mean()

        # Update the actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
