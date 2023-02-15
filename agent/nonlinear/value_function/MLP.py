#!/usr/bin/env python3

# Import modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import agent.nonlinear.nn_utils as nn_utils


# Class definitions
class V(nn.Module):
    """
    Class V is an MLP for estimating the state value function `v`.
    """
    def __init__(self, num_inputs, hidden_dim, init, activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of input feature vector
        hidden_dim : int
            The number of units in each hidden layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : str
            The activation function to use; one of 'relu', 'tanh'
        """
        super(V, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the value of
        `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The feature vector of the state to compute the value of

        Returns
        -------
        torch.Tensor of float
            The value of the state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x


class DiscreteQ(nn.Module):
    """
    Class DiscreteQ implements an action value network with number of
    predicted action values equal to the number of available actions.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init,
                 activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of state feature vector
        num_actions : int
            Dimensionality of the action feature vector
        hidden_dim : int
            The number of units in each hidden layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : str
            The activation function to use; one of 'relu', 'tanh'
        """
        super(DiscreteQ, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through each network, predicting the
        action-value for `action` in `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The state that the action was taken in

        Returns
        -------
        torch.Tensor
            The action value predictions
        """

        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        return self.linear3(x)


class Q(nn.Module):
    """
    Class Q implements an action-value network using an MLP function
    approximator. The action value is computed by concatenating the action to
    the state observation as the input to the neural network.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init,
                 activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of state feature vector
        num_actions : int
            Dimensionality of the action feature vector
        hidden_dim : int
            The number of units in each hidden layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : str
            The activation function to use; one of 'relu', 'tanh'
        """
        super(Q, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state, action):
        """
        Performs the forward pass through each network, predicting the
        action-value for `action` in `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The state that the action was taken in
        action : torch.Tensor of float
            The action taken in the input state to predict the value function
            of

        Returns
        -------
        torch.Tensor
            The action value prediction
        """
        xu = torch.cat([state, action], 1)

        x = self.act(self.linear1(xu))
        x = self.act(self.linear2(x))
        x = self.linear3(x)

        return x


class DoubleQ(nn.Module):
    """
    Class DoubleQ implements two action-value networks,
    computing the action-value function using two separate fully
    connected neural net. This is useful for implementing double Q-learning.
    The action values are computed by concatenating the action to the state
    observation and using this as input to each neural network.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init,
                 activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of state feature vector
        num_actions : int
            Dimensionality of the action feature vector
        hidden_dim : int
            The number of units in each hidden layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : str
            The activation function to use; one of 'relu', 'tanh'
        """
        super(DoubleQ, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state, action):
        """
        Performs the forward pass through each network, predicting two
        action-values (from each action-value approximator) for the input
        action in the input state.

        Parameters
        ----------
        state : torch.Tensor of float
            The state that the action was taken in
        action : torch.Tensor of float
            The action taken in the input state to predict the value function
            of

        Returns
        -------
        2-tuple of torch.Tensor of float
            A 2-tuple of action values, one predicted by each function
            approximator
        """
        xu = torch.cat([state, action], 1)

        x1 = self.act(self.linear1(xu))
        x1 = self.act(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.act(self.linear4(xu))
        x2 = self.act(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
