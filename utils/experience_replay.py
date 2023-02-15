# Import modules
import numpy as np
import torch
from abc import ABC, abstractmethod


# Class definitions
class ExperienceReplay(ABC):
    """
    Abstract base class ExperienceReplay implements an experience replay
    buffer. The specific kind of buffer is determined by classes which
    implement this base class. For example, NumpyBuffer stores all
    transitions in a numpy array while TorchBuffer implements the buffer
    as a torch tensor.

    Attributes
    ----------
    self.cast : func
        A function which will cast data into an appropriate form to be
        stored in the replay buffer. All incoming data is assumed to be
        a numpy array.
    """
    def __init__(self, capacity, seed, state_size, action_size,
                 device=None):
        """
        Constructor

        Parameters
        ----------
        capacity : int
            The capacity of the buffer
        seed : int
            The random seed used for sampling from the buffer
        state_size : tuple[int]
            The number of dimensions of the state features
        action_size : int
            The number of dimensions in the action vector
        """
        self.device = device
        self.is_full = False
        self.position = 0
        self.capacity = capacity

        # Set the casting function, which is needed for implementations which
        # may keep the ER buffer as a different data structure, for example
        # a torch tensor, in this case all data needs to be cast to a torch
        # tensor before storing
        self.cast = lambda x: x

        # Set the random number generator
        self.random = np.random.default_rng(seed=seed)

        # Save the size of states and actions
        self.state_size = state_size
        self.action_size = action_size

        self._sampleable = False

        # Buffer of state, action, reward, next_state, done
        self.state_buffer = None
        self.action_buffer = None
        self.reward_buffer = None
        self.next_state_buffer = None
        self.done_buffer = None
        self.init_buffer()

    @property
    def sampleable(self):
        return self._sampleable

    @abstractmethod
    def init_buffer(self):
        """
        Initializes the buffers on which to store transitions.

        Note that different classes which implement this abstract base class
        may use different data types as buffers. For example, NumpyBuffer
        stores all transitions using a numpy array, while TorchBuffer
        stores all transitions on a torch Tensor on a specific device in order
        to speed up training by keeping transitions on the same device as
        the device which holds the model.

        Post-Condition
        --------------
        The replay buffer self.buffer has been initialized
        """
        pass

    def push(self, state, action, reward, next_state, done):
        """
        Pushes a trajectory onto the replay buffer

        Parameters
        ----------
        state : array_like
            The state observation
        action : array_like
            The action taken by the agent in the state
        reward : float
            The reward seen after taking the argument action in the argument
            state
        next_state : array_like
            The next state transitioned to
        done : bool
            Whether or not the transition was a transition to a goal state
        """
        reward = np.array([reward])
        done = np.array([done])

        state = self.cast(state)
        action = self.cast(action)
        reward = self.cast(reward)
        next_state = self.cast(next_state)
        done = self.cast(done)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        if self.position >= self.capacity - 1:
            self.is_full = True
        self.position = (self.position + 1) % self.capacity
        self._sampleable = False

    @property
    def sampleable(self):
        return self._sampleable

    def is_sampleable(self, batch_size):
        if self.position < batch_size and not self.sampleable:
            return False
        elif not self._sampleable:
            self._sampleable = True

        return self.sampleable

    def sample(self, batch_size):
        """
        Samples a random batch from the buffer

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample

        Returns
        -------
        5-tuple of torch.Tensor
            The arrays of state, action, reward, next_state, and done from the
            batch
        """
        if not self.is_sampleable(batch_size):
            return None, None, None, None, None

        # Get the indices for the batch
        if self.is_full:
            indices = self.random.integers(low=0, high=len(self),
                                           size=batch_size)
        else:
            indices = self.random.integers(low=0, high=self.position,
                                           size=batch_size)

        state = self.state_buffer[indices, :]
        action = self.action_buffer[indices, :]
        reward = self.reward_buffer[indices]
        next_state = self.next_state_buffer[indices, :]
        done = self.done_buffer[indices]

        return state, action, reward, next_state, done

    def __len__(self):
        """
        Gets the number of elements in the buffer

        Returns
        -------
        int
            The number of elements currently in the buffer
        """
        if not self.is_full:
            return self.position
        else:
            return self.capacity


class NumpyBuffer(ExperienceReplay):
    """
    Class NumpyBuffer implements an experience replay buffer. This
    class stores all states, actions, and rewards as numpy arrays.
    For an implementation that uses PyTorch tensors, see
    TorchExperienceReplay
    """
    def __init__(self, capacity, seed, state_size, action_size,
                 state_dtype=np.int32, action_dtype=np.int32):
        """
        Constructor

        Parameters
        ----------
        capacity : int
            The capacity of the buffer
        seed : int
            The random seed used for sampling from the buffer
        state_size : tuple[int]
            The dimensions of the state features
        action_size : int
            The number of dimensions in the action vector
        """
        self._state_dtype = state_dtype
        self._action_dtype = action_dtype
        super().__init__(capacity, seed, state_size, action_size, None)

    def init_buffer(self):
        self.state_buffer = np.zeros((self.capacity, *self.state_size),
                                     dtype=self._state_dtype)
        self.next_state_buffer = np.zeros((self.capacity, *self.state_size),
                                          dtype=self._state_dtype)
        self.action_buffer = np.zeros((self.capacity, self.action_size),
                                      dtype=self._state_dtype)
        self.reward_buffer = np.zeros((self.capacity, 1))
        self.done_buffer = np.zeros((self.capacity, 1), dtype=bool)


class TorchBuffer(ExperienceReplay):
    """
    Class TorchBuffer implements an experience replay buffer. The
    difference between this class and the ExperienceReplay class is that this
    class keeps all experiences as a torch Tensor on the appropriate device
    so that if using PyTorch, we do not need to cast the batch to a
    FloatTensor every time we sample and then place it on the appropriate
    device, as this is very time consuming. This class is basically a
    PyTorch efficient implementation of ExperienceReplay.
    """
    def __init__(self, capacity, seed, state_size, action_size, device):
        """
        Constructor

        Parameters
        ----------
        capacity : int
            The capacity of the buffer
        seed : int
            The random seed used for sampling from the buffer
        device : torch.device
            The device on which the buffer instances should be stored
        state_size : int
            The number of dimensions in the state feature vector
        action_size : int
            The number of dimensions in the action vector
        """
        super().__init__(capacity, seed, state_size, action_size, device)
        self.cast = torch.from_numpy

    def init_buffer(self):
        self.state_buffer = torch.FloatTensor(self.capacity, *self.state_size)
        self.state_buffer = self.state_buffer.to(self.device)

        self.next_state_buffer = torch.FloatTensor(self.capacity,
                                                   *self.state_size)
        self.next_state_buffer = self.next_state_buffer.to(self.device)

        self.action_buffer = torch.FloatTensor(self.capacity, self.action_size)
        self.action_buffer = self.action_buffer.to(self.device)

        self.reward_buffer = torch.FloatTensor(self.capacity, 1)
        self.reward_buffer = self.reward_buffer.to(self.device)

        self.done_buffer = torch.FloatTensor(self.capacity, 1)
        self.done_buffer = self.done_buffer.to(self.device)
