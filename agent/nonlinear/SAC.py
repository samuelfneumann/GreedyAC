# Import modules
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.MLP import SquashedGaussian, Gaussian
from agent.nonlinear.value_function.MLP import DoubleQ, Q
from utils.experience_replay import TorchBuffer as ExperienceReplay
import inspect


class SAC(BaseAgent):
    """
    SAC implements the Soft Actor-Critic algorithm for continuous action spaces
    as found in the paper https://arxiv.org/pdf/1812.05905.pdf.
    """
    def __init__(
        self,
        gamma,
        tau,
        alpha,
        policy,
        target_update_interval,
        critic_lr,
        actor_lr_scale,
        alpha_lr,
        actor_hidden_dim,
        critic_hidden_dim,
        replay_capacity,
        seed,
        batch_size,
        betas,
        env,
        baseline_actions=-1,
        reparameterized=True,
        soft_q=True,
        double_q=True,
        num_samples=1,
        automatic_entropy_tuning=False,
        cuda=False,
        clip_stddev=1000,
        init=None,
        activation="relu",
    ):
        """
        Constructor

        Parameters
        ----------
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
        alpha_lr : float
            The learning rate for the entropy parameter, if using an automatic
            entropy tuning algorithm (see automatic_entropy_tuning) parameter
            below
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
        automatic_entropy_tuning : bool, optional
            Whether the agent should automatically tune its entropy
            hyperparmeter alpha, by default False
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
        soft_q : bool
            Whether or not to learn soft Q functions, by default True. The
            original SAC uses soft Q functions since we learn an
            entropy-regularized policy. When learning an entropy regularized
            policy, guaranteed policy improvement (in the ideal case) only
            exists with respect to soft action values.
        reparameterized : bool
            Whether to use the reparameterization trick to learn the policy or
            to use the log-likelihood trick. The original SAC uses the
            reparameterization trick.
        double_q : bool
            Whether or not to use a double Q critic, by default True
        num_samples : int
            The number of samples to use to estimate the gradient when using a
            likelihood-based SAC (i.e. `reparameterized == False`), by default
            1.

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        super().__init__()
        self._env = env

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        if reparameterized and num_samples != 1:
            raise ValueError

        action_space = env.action_space
        self._action_space = action_space
        obs_space = env.observation_space
        self._obs_space = obs_space
        if len(obs_space.shape) != 1:
            raise ValueError("SAC only supports vector observations")

        self._baseline_actions = baseline_actions

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks.
        self._torch_rng = torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        # Random hypers and fields
        self._is_training = True
        self._gamma = gamma
        self._tau = tau
        self._reparameterized = reparameterized
        self._soft_q = soft_q
        self._double_q = double_q
        if num_samples < 1:
            raise ValueError("cannot have num_samples < 1")
        self._num_samples = num_samples  # Sample for likelihood-based gradient

        self._device = torch.device("cuda:0" if cuda and
                                    torch.cuda.is_available() else "cpu")

        # Experience replay buffer
        self._batch_size = batch_size
        self._replay = ExperienceReplay(replay_capacity, seed, obs_space.shape,
                                        action_space.shape[0], self._device)

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self._target_update_interval = target_update_interval
        self._update_number = 0

        # Automatic entropy tuning
        self._automatic_entropy_tuning = automatic_entropy_tuning
        self._alpha_lr = alpha_lr

        if self._automatic_entropy_tuning and self._alpha_lr <= 0:
            raise ValueError("should not use entropy lr <= 0")

        # Set up the critic and target critic
        self._init_critic(
            obs_space,
            action_space,
            critic_hidden_dim,
            init,
            activation,
            critic_lr,
            betas,
        )

        # Set up the policy
        self._policy_type = policy.lower()
        actor_lr = actor_lr_scale * critic_lr
        self._init_policy(
            obs_space,
            action_space,
            actor_hidden_dim,
            init,
            activation,
            actor_lr,
            betas,
            clip_stddev,
        )

        # Set up auto entropy tuning
        if self._automatic_entropy_tuning:
            self._target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self._device,
            )
            self._alpha = self._log_alpha.exp().detach()
            self._alpha_optim = Adam([self._log_alpha], lr=self._alpha_lr)
        else:
            self._alpha = alpha  # Entropy scale

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info["source"] = source

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        if self._is_training:
            action = self._policy.rsample(state)[0]
        else:
            action = self._policy.rsample(state)[3]

        return action.detach().cpu().numpy()[0]

    def update(self, state, action, reward, next_state, done_mask):
        # Keep transition in replay buffer
        self._replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self._replay.sample(batch_size=self._batch_size)

        if state_batch is None:
            return

        self._update_critic(state_batch, action_batch, reward_batch,
                            next_state_batch, mask_batch)

        self._update_actor(state_batch, action_batch, reward_batch,
                           next_state_batch, mask_batch)

    def _update_actor(self, state_batch, action_batch, reward_batch,
                      next_state_batch, mask_batch):
        """
        Update the actor given a batch of transitions sampled from a replay
        buffer.
        """
        # Calculate the actor loss
        if self._reparameterized:
            # Reparameterization trick
            if self._baseline_actions > 0:
                pi, log_pi = self._policy.rsample(
                    state_batch,
                    num_samples=self._baseline_actions+1,
                )[:2]
                pi = pi.transpose(0, 1).reshape(
                    -1,
                    self._action_space.high.shape[0],
                )
                s_state_batch = state_batch.repeat_interleave(
                    self._baseline_actions + 1,
                    dim=0,
                )
                q = self._get_q(s_state_batch, pi)
                q = q.reshape(self._batch_size, self._baseline_actions + 1, -1)

                # Don't backprop through the approximate state-value baseline
                baseline = q[:, 1:].mean(axis=1).squeeze().detach()

                log_pi = log_pi[0, :, 0]
                q = q[:, 0, 0]
                q -= baseline
            else:
                pi, log_pi = self._policy.rsample(state_batch)[:2]
                q = self._get_q(state_batch, pi)

            policy_loss = ((self._alpha * log_pi) - q).mean()

        else:
            # Log likelihood trick
            baseline = 0
            if self._baseline_actions > 0:
                with torch.no_grad():
                    pi = self._policy.sample(
                        state_batch,
                        num_samples=self._baseline_actions,
                    )[0]
                    pi = pi.transpose(0, 1).reshape(
                        -1,
                        self._action_space.high.shape[0],
                    )
                    s_state_batch = state_batch.repeat_interleave(
                        self._baseline_actions,
                        dim=0,
                    )
                    q = self._get_q(s_state_batch, pi)
                    q = q.reshape(
                        self._batch_size,
                        self._baseline_actions,
                        -1,
                    )
                    baseline = q[:, 1:].mean(axis=1)

            sample = self._policy.sample(
                state_batch,
                self._num_samples,
            )
            pi, log_pi = sample[:2]  # log_pi is differentiable

            if self._num_samples > 1:
                pi = pi.reshape(self._num_samples * self._batch_size, -1)
                state_batch = state_batch.repeat(self._num_samples, 1)

            with torch.no_grad():
                # Context manager ensures that we don't backprop through the q
                # function when minimizing the policy loss
                q = self._get_q(state_batch, pi)
                q -= baseline

            # Compute the policy loss
            log_pi = log_pi.reshape(self._num_samples * self._batch_size, -1)

            with torch.no_grad():
                scale = self._alpha * log_pi - q
            policy_loss = log_pi * scale
            policy_loss = policy_loss.mean()

        # Update the actor
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        # Tune the entropy if appropriate
        if self._automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha *
                           (log_pi + self._target_entropy).detach()).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp().detach()

    def reset(self):
        pass

    def eval(self):
        self._is_training = False

    def train(self):
        self._is_training = True

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        pass

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        pass

    def get_parameters(self):
        pass

    def _init_critic(self, obs_space, action_space, critic_hidden_dim, init,
                     activation, critic_lr, betas):
        """
        Initializes the critic
        """
        num_inputs = obs_space.shape[0]

        if self._double_q:
            critic_type = DoubleQ
        else:
            critic_type = Q

        self._critic = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(device=self._device)

        self._critic_target = critic_type(
            num_inputs,
            action_space.shape[0],
            critic_hidden_dim,
            init,
            activation,
        ).to(self._device)

        # Ensure critic and target critic share the same parameters at the
        # beginning of training
        nn_utils.hard_update(self._critic_target, self._critic)

        self._critic_optim = Adam(
            self._critic.parameters(),
            lr=critic_lr,
            betas=betas,
        )

    def _init_policy(self, obs_space, action_space, actor_hidden_dim, init,
                     activation,  actor_lr, betas, clip_stddev):
        """
        Initializes the policy
        """
        num_inputs = obs_space.shape[0]

        if self._policy_type == "squashedgaussian":
            self._policy = SquashedGaussian(num_inputs, action_space.shape[0],
                                            actor_hidden_dim, activation,
                                            action_space, clip_stddev,
                                            init).to(self._device)

        elif self._policy_type == "gaussian":
            self._policy = Gaussian(num_inputs, action_space.shape[0],
                                    actor_hidden_dim, activation, action_space,
                                    clip_stddev, init).to(self._device)

        else:
            raise NotImplementedError(f"policy {self._policy_type} unknown")

        self._policy_optim = Adam(
            self._policy.parameters(),
            lr=actor_lr,
            betas=betas,
        )

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
        if self._double_q:
            q1, q2 = self._critic(state_batch, action_batch)
            return torch.min(q1, q2)
        else:
            return self._critic(state_batch, action_batch)

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        """
        Update the critic(s) given a batch of transitions sampled from a replay
        buffer.
        """
        if self._double_q:
            self._update_double_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        else:
            self._update_single_critic(state_batch, action_batch, reward_batch,
                                       next_state_batch, mask_batch)

        # Increment the running total of updates and update the critic target
        # if needed
        self._update_number += 1
        if self._update_number % self._target_update_interval == 0:
            self._update_number = 0
            nn_utils.soft_update(self._critic_target, self._critic, self._tau)

    def _update_single_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a single Q
        critic.
        """
        if self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi = \
                self._policy.sample(next_state_batch)[:2]

            if len(next_state_log_pi.shape) == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)

            # Calculate the Q value of the next action in the next state
            q_next = self._critic_target(next_state_batch,
                                         next_state_action)

            if self._soft_q:
                q_next -= self._alpha * next_state_log_pi

            # Calculate the target for the SARSA update
            q_target = reward_batch + mask_batch * self._gamma * q_next

        # Calculate the Q value of each action in each respective state
        q = self._critic(state_batch, action_batch)

        # Calculate the loss between the target and estimate Q values
        q_loss = F.mse_loss(q, q_target)

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()

    def _update_double_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a double Q
        critic.
        """

        if not self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi = \
                self._policy.sample(next_state_batch)[:2]

            # Calculate the action values for the next state
            next_q1, next_q2 = self._critic_target(next_state_batch,
                                                   next_state_action)

            # Double Q: target uses the minimum of the two computed action
            # values
            min_next_q = torch.min(next_q1, next_q2)

            # If using soft action value functions, then adjust the target
            if self._soft_q:
                min_next_q -= self._alpha * next_state_log_pi

            # Calculate the target for the action value function update
            q_target = reward_batch + mask_batch * self._gamma * min_next_q

        # Calculate the two Q values of each action in each respective state
        q1, q2 = self._critic(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()
