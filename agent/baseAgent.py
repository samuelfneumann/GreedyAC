from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Class BaseAgent implements the base functionality for all agents

    Attributes
    ----------
    self.batch : bool
        Whether or not the agent is using batch updates, by default False.
    self.info : dict
        A dictionary which records agent info
    """
    def __init__(self):
        """
        Constructor
        """
        self.batch = False
        self.info = {}

    """
    BaseAgent is the abstract base class for all agents
    """
    @abstractmethod
    def sample_action(self, state):
        """
        Samples an action from the agent

        Parameters
        ----------
        state : np.array
            The state feature vector

        Returns
        -------
        array_like of float
            The action to take
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step, which may be a number of offline
        batch updates

        Parameters
        ----------
        state : np.array or array_like of np.array
            The state feature vector
        action : np.array of float or array_like of np.array
            The action taken
        reward : float or array_like of float
            The reward seen by the agent after taking the action
        next_state : np.array or array_like of np.array
            The feature vector of the next state transitioned to after the
            agent took the argument action
        done_mask : bool or array_like of bool
            False if the agent reached the goal, True if the agent did not
            reach the goal yet the episode ended (e.g. max number of steps
            reached)

        Return
        ------
        4-tuple of array_like
            A tuple containing array_like, each of which contains the states,
            actions, rewards, and next states used in the update
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the agent between episodes
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        pass

    @abstractmethod
    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """
        Gets all learned agent parameters such that training can be resumed.

        Gets all parameters of the agent such that, if given the
        hyperparameters of the agent, training is resumable from this exact
        point. This include the learned average reward, the learned entropy,
        and other such learned values if applicable. This does not only apply
        to the weights of the agent, but *all* values that have been learned
        or calculated during training such that, given these values, training
        can be resumed from this exact point.

        For example, in the LinearAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to int, float, array_like, and/or torch.Tensor
            The agent's weights
        """
        pass
