#!/usr/bin/env python3

# Adapted from OpenAI Gym Pendulum-v0

# Import modules
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    """
    PendulumEnv is a modified version of the Pendulum-v0 OpenAI Gym
    environment. In this version, the reward is the cosine of the angle
    between the pendulum and its fixed base. The angle is measured vertically
    so that if the pendulum stays straight up, the angle is 0 radians, and
    if the pendulum points straight down, then the angle is π raidans.
    Therefore, the agent will get reward cos(0) = 1 if the pendulum stays
    straight up and reward of cos(π) = -1 if the pendulum stays straight
    down. The goal is to have the pendulum stay straight up as long as
    possible.

    In this version of the Pendulum environment, state features may either
    be encoded as the cosine and sine of the pendulums angle with respect to
    it fixed base (reference axis vertical above the base) and the angular
    velocity, or as the angle itself and the angular velocity. If θ is the
    angle between the pendulum and the positive y-axis (axis straight up above
    the base) and ω is the angular velocity, then the states may be encoded
    as [cos(θ), sin(θ), ω] or as [θ, ω] depending on the argument trig_features
    to the constructor. The encoding [cos(θ), sin(θ), ω] is a somewhat easier
    problem, since cos(θ) is exactly the reward seen in that state.

    Let θ be the angle of the pendulum with respect to the vertical axis from
    the pendulum's base, ω be the angular velocity, and τ be the torque
    applied to the base. Then:
        1. State features are vectors: [cos(θ), sin(θ), ω] if the
           self.trig_features variable is True, else [θ, ω]
        2. Actions are 1-dimensional vectors that denote the torque applied
           to the pendulum's base: τ ∈ [-2, 2]
        3. Reward is the cosine of the pendulum with respect to the fixed
           base, measured with respect to the vertical axis proceeding above
           the pendulum's base: cos(θ)
        4. The start state is always with the pendulum horizontal, pointing to
           the right, with 0 angular velocity

    Note that this is a continuing task.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, continuous_action=True, g=10.0, trig_features=False,
            seed=None):
        """
        Constructor

        Parameters
        ----------
        g : float, optional
            Gravity, by default 10.0
        trig_features : bool
            Whether to use trigonometric encodings of features or to use the
            angle itself, by default False. If True, then state features are
            [cos(θ), sin(θ), ω], else state features are [θ, ω] (see class
            documentation)
        seed : int
            The seed with which to seed the environment, by default None

        """
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.length = 1.
        self.viewer = None
        self.continuous_action = continuous_action

        # Set the actions
        if self.continuous_action:
            self.action_space = spaces.Box(
                low=-self.max_torque,
                high=self.max_torque, shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(3)

        # Set the states
        self.trig_features = trig_features
        if trig_features:
            # Encode states as [cos(θ), sin(θ), ω]
            high = np.array([1., 1., self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-high,
                high=high,
                dtype=np.float32
            )
        else:
            # Encode states as [θ, ω]
            low = np.array([-np.pi, -self.max_speed], dtype=np.float32)
            high = np.array([np.pi, self.max_speed], dtype=np.float32)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32
            )

        self.seed(seed)

    def seed(self, seed=None):
        """
        Sets the random seed for the environment

        Parameters
        ----------
        seed : int, optional
            The random seed for the environment, by default None

        Returns
        -------
        list
            The random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        """
        Takes a single environmental step

        Parameters
        ----------
        u : array_like of float
            The torque to apply to the base of the pendulum

        Returns
        -------
        3-tuple of array_like, float, bool, dict
            The state observation, the reward, the done flag (always False),
            and some info about the step
        """
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        length = self.length
        dt = self.dt

        if self.continuous_action:
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
        else:
            assert self.action_space.contains(u), \
                f"{action!r} ({type(action)}) invalid"
            u = (u - 1) * self.max_torque  # [-max_torque, 0, max_torque]

        self.last_u = u  # for rendering

        newthdot = thdot + (-3 * g / (2 * length) * np.sin(th + np.pi) + 3. /
                            (m * length ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        reward = np.cos(newth)

        if self.trig_features:
            # States are encoded as [cos(θ), sin(θ), ω]
            return self._get_obs(), reward, False, {}

        # States are encoded as [θ, ω]
        return self.state, reward, False, {}

    def reset(self):
        """
        Resets the environment to its starting state

        Returns
        -------
        array_like of float
            The starting state feature representation
        """
        state = np.array([np.pi, 0.])
        self.state = angle_normalize(state)
        # self.state = start
        self.last_u = None

        if self.trig_features:
            # States are encoded as [cos(θ), sin(θ), ω]
            return self._get_obs()

        # States are encoded as [θ, ω]
        return self.state

    def _get_obs(self):
        """
        Creates and returns the state feature vector

        Returns
        -------
        array_like of float
            The state feature vector
        """
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        """
        Renders the current time frame

        Parameters
        ----------
        mode : str, optional
            Which mode to render in, by default 'human'

        Returns
        -------
        array_like
            The image of the current time step
        """
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
        Closes the viewer
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    """
    Normalizes the input angle to the range [-π, π]

    Parameters
    ----------
    x : float
        The angle to normalize

    Returns
    -------
    float
        The normalized angle
    """
    # return x % (2 * np.pi)
    return (((x+np.pi) % (2*np.pi)) - np.pi)
