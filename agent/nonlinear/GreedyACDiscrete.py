# Import modules
from gym.spaces import Box, Discrete
import inspect
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from agent.baseAgent import BaseAgent
from utils.experience_replay import TorchBuffer as ExperienceReplay
from agent.nonlinear.value_function.MLP import Q as QMLP
from agent.nonlinear.policy.MLP import Softmax
import agent.nonlinear.nn_utils as nn_utils


class GreedyACDiscrete(BaseAgent):
    """
    GreedyACDiscrete implements the GreedyAC algorithm with discrete actions
    """
    def __init__(self, num_inputs, action_space, gamma, tau, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, betas, cuda=False,
                 clip_stddev=1000, init=None, entropy_from_single_sample=True,
                 activation="relu"):
        super().__init__()

        self.batch = True

        # The number of top actions to increase the probability of taking
        self.top_actions = 1

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
        self.entropy_from_single_sample = entropy_from_single_sample
        self.gamma = gamma
        self.tau = tau  # Polyak average
        self.state_dims = num_inputs

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Discrete):
            self.action_dims = 1
            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed,
                                           (num_inputs,), 1, self.device)
        else:
            raise ValueError("GreedyACDiscrete must use discrete action")

        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        # Create the critic Q function
        if isinstance(action_space, Box):
            raise ValueError("GreedyACDiscrete must use discrete actions")
        elif isinstance(action_space, Discrete):
            action_shape = 1

        self.critic = QMLP(num_inputs, action_shape,
                           critic_hidden_dim, init, activation).to(
                               device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr,
                                 betas=betas)

        self.critic_target = QMLP(num_inputs, action_shape,
                                  critic_hidden_dim, init, activation).to(
                                      self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self._create_policies(policy, num_inputs, action_space,
                              actor_hidden_dim, clip_stddev, init, activation)

        actor_lr = actor_lr_scale * critic_lr
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                 betas=betas)

        self.is_training = True

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {}
        self.info = {
            "action_values": [],
            "source": source,
        }

    def update(self, state, action, reward, next_state, done_mask):
        # Adjust action shape to ensure it fits in replay buffer properly
        action = np.array([action])

        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        if state_batch is None:
            # Not enough samples in buffer
            return

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, _, _ = \
                self.policy.sample(next_state_batch)

            next_q = self.critic_target(next_state_batch, next_state_action)
            target_q_value = reward_batch + mask_batch * self.gamma * next_q

        q_value = self.critic(state_batch, action_batch)

        # Calculate the loss on the critic
        q_loss = F.mse_loss(target_q_value, q_value)

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Update target networks
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

        # Sample actions from the sampler to determine which to update
        # with
        with torch.no_grad():
            action_batch = self.sampler(state_batch)
        stacked_s_batch = state_batch.repeat_interleave(self.num_actions,
                                                        dim=0)

        # Get the values of the sampled actions and find the best
        # self.top_actions actions
        with torch.no_grad():
            q_values = self.critic(stacked_s_batch, action_batch)

        q_values = q_values.reshape(self.batch_size, self.num_actions,
                                    1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :self.top_actions]
        best_ind = best_ind.repeat_interleave(self.action_dims, -1)

        action_batch = action_batch.reshape(self.batch_size, self.num_actions,
                                            self.action_dims)
        best_actions = torch.gather(action_batch, 1, best_ind)

        # Reshape samples for calculating the loss
        stacked_s_batch = state_batch.repeat_interleave(self.top_actions,
                                                        dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dims))

        # Actor loss
        # print(stacked_s_batch.shape, best_actions.shape)
        # print("Computing actor loss")
        policy_loss = self.policy.log_prob(stacked_s_batch, best_actions)
        policy_loss = -policy_loss.mean()

        # Update actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

    def sample_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        act = action.detach().cpu().numpy()[0][0]

        return act

    def reset(self):
        pass

    def eval(self):
        self.is_training = False

    def train(self):
        self.is_training = True

    def _create_policies(self, policy, num_inputs, action_space,
                         actor_hidden_dim, clip_stddev, init, activation):
        self.policy_type = policy.lower()
        if self.policy_type == "softmax":
            self.num_actions = action_space.n
            self.policy = Softmax(num_inputs, self.num_actions,
                                  actor_hidden_dim, activation,
                                  init).to(self.device)

            # Sampler returns every available action in each state
            def sample(state_batch):
                batch_size = state_batch.shape[0]
                actions = torch.tensor([n for n in range(self.num_actions)])
                actions = actions.repeat(batch_size).unsqueeze(-1)

                return actions

            self.sampler = sample

        else:
            raise NotImplementedError

    def get_parameters(self):
        pass

    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        pass

    def load_model(self, actor_path, critic_path):
        pass
