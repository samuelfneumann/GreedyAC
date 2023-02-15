# Import modules
import torch
import inspect
from gym.spaces import Box, Discrete
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.MLP import Gaussian
from agent.nonlinear.value_function.MLP import Q as QMLP
from utils.experience_replay import TorchBuffer as ExperienceReplay


class VAC(BaseAgent):
    """
    VAC implements the Vanilla Actor-Critic agent
    """

    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 num_samples, actor_hidden_dim, critic_hidden_dim,
                 replay_capacity, seed, batch_size, betas, env, cuda=False,
                 clip_stddev=1000, init=None, activation="relu",
                 reparameterized=False):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of input features
        action_space : gym.spaces.Space
            The action space from the gym environment
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
        super().__init__()
        self.batch = True

        self._t = -1

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
        self.action_space = action_space
        self.reparam = reparameterized

        if not isinstance(action_space, Box):
            raise ValueError("VAC only works with Box action spaces")

        self.state_dims = num_inputs
        self.num_samples = num_samples - 1
        if not self.reparam:
            assert num_samples >= 2

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Box):
            self.action_dims = action_space.high.shape[0]

            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           (num_inputs,),
                                           action_space.shape[0], self.device)
        elif isinstance(action_space, Discrete):
            self.action_dims = 1
            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed, num_inputs,
                                           1, self.device)
        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        # Create the critic Q function
        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1

        self.critic = QMLP(num_inputs, action_shape, critic_hidden_dim,
                           init, activation).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr,
                                 betas=betas)

        self.critic_target = QMLP(num_inputs, action_shape,
                                  critic_hidden_dim, init, activation).to(
                                      self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self.policy_type = policy.lower()
        actor_lr = actor_lr_scale * critic_lr
        if self.policy_type == "gaussian":

            self.policy = Gaussian(num_inputs, action_space.shape[0],
                                   actor_hidden_dim, activation,
                                   action_space, clip_stddev, init).to(
                                       self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                     betas=betas)

        else:
            raise NotImplementedError

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info["source"]: source

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        act = action.detach().cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done_mask):
        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            return

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, _, _ = \
                self.policy.sample(next_state_batch)
            qf_next_value = self.critic_target(next_state_batch,
                                               next_state_action)

            q_target = reward_batch + mask_batch * self.gamma * qf_next_value

        q_prediction = self.critic(state_batch, action_batch)
        q_loss = F.mse_loss(q_prediction, q_target)

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        if not self.reparam:
            # Sample action that the agent would take
            pi, _, _ = self.policy.sample(state_batch)

            # Calculate the advantage
            with torch.no_grad():
                q_pi = self.critic(state_batch, pi)
            sampled_actions, _, _ = self.policy.sample(state_batch,
                                                       self.num_samples)
            if self.num_samples == 1:
                sampled_actions = sampled_actions.unsqueeze(0)
            sampled_actions = torch.permute(sampled_actions, (1, 0, 2))

            state_baseline = 0
            if self.num_samples > 2:
                # Baseline computed with self.num_samples - 1 action
                # value estimates
                baseline_actions = sampled_actions
                baseline_actions = torch.reshape(baseline_actions,
                                                 [-1, self.action_dims])
                stacked_s_batch = torch.repeat_interleave(state_batch,
                                                          self.num_samples,
                                                          dim=0)
                stacked_s_batch = torch.reshape(stacked_s_batch,
                                                [-1, self.state_dims])

                baseline_q_vals = self.critic(stacked_s_batch,
                                              baseline_actions)

                baseline_q_vals = torch.reshape(baseline_q_vals,
                                                [self.batch_size,
                                                    self.num_samples])
                state_baseline = baseline_q_vals.mean(axis=1).unsqueeze(1)
            advantage = q_pi - state_baseline

            # Estimate the entropy from a single sampled action in each state
            entropy_actions = pi
            entropy = self.policy.log_prob(state_batch, entropy_actions)
            with torch.no_grad():
                entropy *= entropy
            entropy = -entropy

            policy_loss = self.policy.log_prob(state_batch, pi) * advantage
            policy_loss = -(policy_loss + (self.alpha * entropy)).mean()
        else:
            if self.num_samples >= 1:
                pi, log_pi = self.policy.rsample(
                    state_batch,
                    num_samples=self.num_samples,
                )[:2]
                pi = pi.transpose(0, 1).reshape(
                    -1,
                    self.action_space.high.shape[0],
                )
                s_state_batch = state_batch.repeat_interleave(
                    self.num_samples,
                    dim=0,
                )
                q = self.critic(s_state_batch, pi)
                q = q.reshape(self.batch_size, self.num_samples + 1, -1)

                # Don't backprop through the approximate state-value baseline
                baseline = q[:, 1:].mean(axis=1).squeeze().detach()

                log_pi = log_pi[0, :, 0]
                q = q[:, 0, 0]
                q -= baseline
            else:
                pi, log_pi = self.policy.rsample(state_batch)[:2]
                q = self.critic(state_batch, pi)

            policy_loss = ((self.alpha * log_pi) - q).mean()

        # Update the actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update target network
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

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
