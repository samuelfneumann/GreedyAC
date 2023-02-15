from copy import deepcopy
import gym
from gym import spaces
from gym.envs import register
import numpy as np

from minatar import Environment


class GymEnv(gym.Env):
    """
    GymEnv wraps MinAtar environments to change their interface to the OpenAI
    Gym interface
    """
    metadata = {"render.modes": ["human", "array"]}

    def __init__(self, game, display_time=50, use_minimal_action_set=True,
                 **kwargs):
        self.game_name = game
        self.display_time = display_time

        self.game_kwargs = kwargs
        self.seed()

        if use_minimal_action_set:
            self._action_set = self.game.minimal_action_set()
        else:
            self._action_set = list(range(self.game.num_actions()))

        self._action_space = spaces.Discrete(len(self._action_set))
        self._observation_space = spaces.Box(
            0.0, 1.0, shape=self.game.state_shape(), dtype=bool
        )

    @property
    def action_space(self):
        """
        Gets the action space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The action space
        """
        return self._action_space

    @property
    def observation_space(self):
        """
        Gets the observation space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The observation space
        """
        return self._observation_space

    def step(self, action):
        action = self._action_set[action]
        reward, done = self.game.act(action)

        return self.game.state(), reward, done, {}

    def reset(self):
        self.game.reset()
        return self.game.state()

    def seed(self, seed=None):
        self.game = Environment(
            env_name=self.game_name,
            random_seed=seed,
            **self.game_kwargs
        )
        return seed

    def render(self, mode="human"):
        if mode == "array":
            return self.game.state()
        elif mode == "human":
            self.game.display_state(self.display_time)


class BatchFirst(gym.Env):
    """
    BatchFirst permutes the axes of state observations for MinAtar environments
    so that the batch dimension is first.
    """
    def __init__(self, env):
        self.env = env

        # Adjust observation space
        obs = deepcopy(self.env.observation_space)
        low = obs.low

        low = np.moveaxis(low, (0, 1, 2), (2, 1, 0))

        self._observation_space = spaces.Box(
            0.0, 1.0, shape=low.shape, dtype=bool
        )

    @property
    def action_space(self):
        """
        Gets the action space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The action space
        """
        return self.env.action_space

    @property
    def observation_space(self):
        """
        Gets the observation space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The observation space
        """
        return self._observation_space

    def seed(self, seed):
        """
        Seeds the environment with a random seed

        Parameters
        ----------
        seed : int
            The random seed to seed the environment with
        """
        self.env.seed(seed)

    def reset(self):
        state = self.env.reset()
        state = np.moveaxis(state, (0, 1, 2), (2, 1, 0))

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        state = np.moveaxis(state, (0, 1, 2), (2, 1, 0))

        print(state.shape)

        return state, reward, done, info

    def render(self, mode="human"):
        return self.env.render(mode)
