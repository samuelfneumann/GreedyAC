# Import modules
import gym
from copy import deepcopy
from env.PendulumEnv import PendulumEnv
from env.Acrobot import AcrobotEnv
import env.MinAtar as MinAtar
import numpy as np


class Environment:
    """
    Class Environment is a wrapper around OpenAI Gym environments, to ensure
    logging can be done as well as to ensure that we can restrict the episode
    time steps.
    """
    def __init__(self, config, seed, monitor=False, monitor_after=0):
        """
        Constructor

        Parameters
        ----------
        config : dict
            The environment configuration file
        seed : int
            The seed to use for all random number generators
        monitor : bool
            Whether or not to render the scenes as the agent learns, by
            default False
        monitor_after : int
            If monitor is True, how many timesteps should pass before
            the scene is rendered, by default 0.
        """
        # Overwrite rewards and start state if necessary
        self.overwrite_rewards = config["overwrite_rewards"]
        self.rewards = config["rewards"]
        self.start_state = np.array(config["start_state"])

        self.steps = 0
        self.episodes = 0

        # Keep track of monitoring
        self.monitor = monitor
        self.steps_until_monitor = monitor_after

        self.env_name = config["env_name"]

        self.env = env_factory(config)
        print("Seeding environment:", seed)
        self.env.seed(seed=seed)
        self.steps_per_episode = config["steps_per_episode"]

        # Increase the episode steps of the wrapped OpenAI gym environment so
        # that this wrapper will timeout before the OpenAI gym one does
        self.env._max_episode_steps = self.steps_per_episode + 10

        if "info" in dir(self.env):
            self.info = self.env.info
        else:
            self.info = {}

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
        return self.env.observation_space

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
        """
        Resets the environment by resetting the step counter to 0 and resetting
        the wrapped environment. This function also increments the total
        episode count.

        Returns
        -------
        2-tuple of array_like, dict
            The new starting state and an info dictionary
        """
        self.steps = 0
        self.episodes += 1

        state = self.env.reset()

        # If the user has inputted a fixed start state, use that instead
        if self.start_state.shape[0] != 0:
            state = self.start_state
            self.env.state = state

        return state, {"orig_state": state}

    def render(self):
        """
        Renders the current frame
        """
        self.env.render()

    def step(self, action):
        """
        Takes a single environmental step

        Parameters
        ----------
        action : array_like of float
            The action array. The number of elements in this array should be
            the same as the action dimension.

        Returns
        -------
        float, array_like of float, bool, dict
            The reward and next state as well as a flag specifying if the
            current episode has been completed and an info dictionary
        """
        if self.monitor and self.steps_until_monitor < 0:
            self.render()

        self.steps += 1
        self.steps_until_monitor -= (1 if self.steps_until_monitor >= 0 else 0)

        # Get the next state, reward, and done flag
        state, reward, done, info = self.env.step(action)
        info["orig_state"] = state

        # If the episode completes, return the goal reward
        if done:
            info["steps_exceeded"] = False
            if self.overwrite_rewards:
                reward = self.rewards["goal"]
            return state, reward, done, info

        # If the user has set rewards per timestep
        if self.overwrite_rewards:
            reward = self.rewards["timestep"]

        # If the maximum time-step was reached
        if self.steps >= self.steps_per_episode > 0:
            done = True
            info["steps_exceeded"] = True

        return state, reward, done, info


def env_factory(config):
    """
    Instantiates and returns an environment given an environment name.

    Parameters
    ----------
    config : dict
        The environment config

    Returns
    -------
    gym.Env
        The environment to train on
    """
    name = config["env_name"]
    seed = config["seed"]
    env = None

    if name == "Pendulum-v0":
        env = PendulumEnv(seed=seed, continuous_action=config["continuous"])

    elif name == "PendulumPenalty-v0":
        env = pp.PendulumEnv(seed=seed, continuous_action=config["continuous"])

    elif name == "PositivePendulumPenalty-v0":
        env = ppp.PendulumEnv(seed=seed,
                              continuous_action=config["continuous"])

    elif name == "PendulumNoShaped-v0":
        env = pens.PendulumEnv(seed=seed,
                               continuous_action=config["continuous"])

    elif name == "PendulumNoShapedPenalty-v0":
        env = pensp.PendulumEnv(seed=seed,
                                continuous_action=config["continuous"])

    elif name == "MountainCarShaped":
        env = mcs.MountainCar()

    elif name == "Bimodal" or name == "Bimodal":
        reward_variance = config.get("reward_variance", True)
        env = Bimodal(seed, reward_variance)

    elif name == "Bandit":
        n_actions = config.get("n_action", 10)
        env = Bandit(seed, n_actions)

    elif name == "ContinuousCartpole-v0":
        env = ContinuousCartPoleEnv()

    elif name == "IndexGridworld":
        env = IndexGridworldEnv(config["rows"], config["cols"])
        env.seed(seed)

    elif name == "XYGridworld":
        env = XYGridworldEnv(config["rows"], config["cols"])
        env.seed(seed)

    elif name == "Gridworld":
        env = GridworldEnv(config["rows"], config["cols"])
        env.seed(seed)

    elif name == "PuddleWorld-v1":
        env = PuddleWorldEnv(continuous=config["continuous"], seed=seed)

    elif name == "Acrobot-v1":
        env = AcrobotEnv(seed=seed, continuous_action=config["continuous"])

    elif name == "CGW":
        env = CGW.GridWorld()

    elif name == "ContinuousGridWorld":
        env = ContinuousGridWorld.GridWorld()

    elif "minatar" in name.lower():
        if "/" in name:
            raise ValueError(f"specify environment as MinAtar{name} rather " +
                             "than MinAtar/{name}")
        minimal_actions = config.get("use_minimal_action_set", True)
        stripped_name = name[7:].lower()  # Strip off "MinAtar"
        env = MinAtar.GymEnv(
            stripped_name,
            use_minimal_action_set=minimal_actions,
        )

    else:
        # Ensure we use the base gym environment. `gym.make` returns a TimeStep
        # environment wrapper, but we want the underlying environment alone.
        env = gym.make(name).env
        env.seed(seed)

    print(config)
    return env
