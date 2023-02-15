#!/usr/bin/env python3

# Adapted from https://github.com/pranz24/pytorch-soft-actor-critic

# Import modules
import torch
import numpy as np
from agent.baseAgent import BaseAgent


class Random(BaseAgent):
    """
    Random implement a random agent, which is one which samples uniformly from
    all available actions.
    """
    def __init__(self, action_space, seed):
        super().__init__()
        self.batch = False

        self.action_dims = len(action_space.high)
        self.action_low = action_space.low
        self.action_high = action_space.high

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.policy = torch.distributions.Uniform(
            torch.Tensor(action_space.low), torch.Tensor(action_space.high))

    def sample_action(self, _):
        """
        Samples an action from the agent

        Parameters
        ----------
        _ : np.array
            The state feature vector

        Returns
        -------
        array_like of float
            The action to take
        """
        action = self.policy.sample()

        return action.detach().cpu().numpy()

    def sample_action_(self, _, size):
        """
        sample_action_ is like sample_action, except the rng for
        action selection in the environment is not affected by running
        this function.
        """
        return self.rng.uniform(self.action_low, self.action_high,
                                size=(size, self.action_dims))

    def update(self, _, _1, _2, _3, _4):
        pass

    def update_value_fn(self, _, _1, _2, _3, _4, _5):
        pass

    def reset(self):
        """
        Resets the agent between episodes
        """
        pass

    def eval(self):
        pass

    def train(self):
        pass

    # Save model parameters
    def save_model(self, _, _1="", _2=None, _3=None):
        pass

    # Load model parameters
    def load_model(self, _, _1):
        pass

    def get_parameters(self):
        pass
