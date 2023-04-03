# Import modules
import numpy as np
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
try:
    import runs
except ModuleNotFoundError:
    import utils.runs


def create_agent(agent, config):
    """
    Creates an agent given the agent name and configuration dictionary

    Parameters
    ----------
    agent : str
        The name of the agent, one of 'linearAC' or 'SAC'
    config : dict
        The agent configuration dictionary

    Returns
    -------
    baseAgent.BaseAgent
        The agent to train
    """
    # Random agent
    if agent.lower() == "random":
        from agent.Random import Random
        return Random(config["action_space"], config["seed"])

    # Vanilla Actor-Critic
    if agent.lower() == "VAC".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.VAC import VAC
        return VAC(
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"], tau=config["tau"],
            alpha=config["alpha"], policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"], batch_size=config["batch_size"],
            cuda=config["cuda"], clip_stddev=config["clip_stddev"],
            init=config["weight_init"], betas=config["betas"],
            num_samples=config["num_samples"], activation="relu",
            env=config["env"],
        )

    # Discrete Vanilla Actor-Critic
    if agent.lower() == "VACDiscrete".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.VACDiscrete import VACDiscrete
        return VACDiscrete(
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"], tau=config["tau"],
            alpha=config["alpha"], policy=config["policy_type"],
            target_update_interval=config[
                "target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"], batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"], betas=config["betas"],
            activation="relu",
        )

    # Soft Actor-Critic
    if agent.lower() == "SAC".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        if "num_hidden" in config:
            num_hidden = config["num_hidden"]
        else:
            num_hidden = 3
        from agent.nonlinear.SAC import SAC
        return SAC(
            baseline_actions=config["baseline_actions"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            alpha_lr=config["alpha_lr"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            automatic_entropy_tuning=config["automatic_entropy_tuning"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"],
            betas=config["betas"],
            activation=activation,
            env=config["env"],
            soft_q=config["soft_q"],
            reparameterized=config["reparameterized"],
            double_q=config["double_q"],
            num_samples=config["num_samples"],
        )

    # Discrete Soft Actor-Critic
    if agent.lower() == "SACDiscrete".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        if "num_hidden" in config:
            num_hidden = config["num_hidden"]
        else:
            num_hidden = 3

        from agent.nonlinear.SACDiscrete import SACDiscrete
        return SACDiscrete(
            env=config["env"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"],
            betas=config["betas"],
            activation=activation,
            double_q=config["double_q"],
            soft_q=config["soft_q"],
        )

    # Discrete GreedyAC
    if agent.lower() == "GreedyACDiscrete".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.GreedyACDiscrete import GreedyACDiscrete
        return GreedyACDiscrete(
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"], tau=config["tau"],
            policy=config["policy_type"],
            target_update_interval=config[
                "target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"],
            betas=config["betas"], activation=activation,
        )

    # GreedyAC
    if agent.lower() == "GreedyAC".lower():
        if "activation" in config:
            activation = config["activation"]
        else:
            activation = "relu"

        from agent.nonlinear.GreedyAC import GreedyAC
        return GreedyAC(
            num_inputs=config["feature_size"],
            action_space=config["action_space"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            policy=config["policy_type"],
            target_update_interval=config["target_update_interval"],
            critic_lr=config["critic_lr"],
            actor_lr_scale=config["actor_lr_scale"],
            actor_hidden_dim=config["hidden_dim"],
            critic_hidden_dim=config["hidden_dim"],
            replay_capacity=config["replay_capacity"],
            seed=config["seed"],
            batch_size=config["batch_size"],
            cuda=config["cuda"],
            clip_stddev=config["clip_stddev"],
            init=config["weight_init"],
            rho=config["n_rho"][1],
            num_samples=config["n_rho"][0],
            betas=config["betas"], activation=activation,
            env=config["env"],
        )

    raise NotImplementedError("No agent " + agent)


def _calculate_mean_return_episodic(hp_returns, type_, after=0):
    """
    Calculates the mean return for an experiment run on an episodic environment
    over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    if type_ == "eval":
        hp_returns = [np.mean(hp_returns[i][after:], axis=-1) for i in
                      range(len(hp_returns))]

    # Calculate the average return for all episodes in the run
    run_returns = [np.mean(hp_returns[i][after:]) for i in
                   range(len(hp_returns))]

    mean = np.mean(run_returns)
    stderr = np.std(run_returns) / np.sqrt(len(hp_returns))

    return mean, stderr


def _calculate_mean_return_episodic_conf(hp_returns, type_, significance,
                                         after=0):
    """
    Calculates the mean return for an experiment run on an episodic environment
    over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    significance: float
        The level of significance for the confidence interval
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    if type_ == "eval":
        hp_returns = [np.mean(hp_returns[i][after:], axis=-1) for i in
                      range(len(hp_returns))]

    # Calculate the average return for all episodes in the run
    run_returns = [np.mean(hp_returns[i][after:]) for i in
                   range(len(hp_returns))]

    mean = np.mean(run_returns)
    run_returns = np.array(run_returns)

    conf = bs.bootstrap(run_returns, stat_func=bs_stats.mean,
                        alpha=significance)

    return mean, conf


def _calculate_mean_return_continuing(hp_returns, type_, after=0):
    """
    Calculates the mean return for an experiment run on a continuing
    environment over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    hp_returns = np.stack(hp_returns)

    # If evaluating, use the mean return over all episodes for each
    # evaluation interval. That is, if 10 eval episodes for each
    # evaluation the take the average return over all these eval
    # episodes
    if type_ == "eval":
        hp_returns = hp_returns.mean(axis=-1)

    # Calculate the average return over all runs
    hp_returns = hp_returns[after:, :].mean(axis=-1)

    # Calculate the average return over all "episodes"
    stderr = np.std(hp_returns) / np.sqrt(len(hp_returns))
    mean = hp_returns.mean(axis=0)

    return mean, stderr


def _calculate_mean_return_continuing_conf(hp_returns, type_, significance,
                                           after=0):
    """
    Calculates the mean return for an experiment run on a continuing
    environment over all runs and episodes

    Parameters
    ----------
    hp_returns : Iterable of Iterable
        A list of lists, where the outer list has a single inner list for each
        run. The inner lists store the return per episode for that run. Note
        that these returns should be for a single hyperparameter setting, as
        everything in these lists are averaged and returned as the average
        return.
    type_ : str
        Whether calculating the training or evaluation mean returns, one of
        'train', 'eval'
    after : int, optional
        Only consider episodes after this episode, by default 0

    Returns
    -------
    2-tuple of float
        The mean and standard error of the returns over all episodes and all
        runs
    """
    hp_returns = np.stack(hp_returns)

    # If evaluating, use the mean return over all episodes for each
    # evaluation interval. That is, if 10 eval episodes for each
    # evaluation the take the average return over all these eval
    # episodes
    if type_ == "eval":
        hp_returns = hp_returns.mean(axis=-1)

    # Calculate the average return over all episodes
    hp_returns = hp_returns[after:, :].mean(axis=-1)

    # Calculate the average return over all runs
    mean = hp_returns.mean(axis=0)
    conf = bs.bootstrap(hp_returns, stat_func=bs_stats.mean,
                        alpha=significance)

    return mean, conf


def combine_runs(data1, data2):
    """
    Adds the runs for each hyperparameter setting in data2 to the runs for the
    corresponding hyperparameter setting in data1.

    Given two data dictionaries, this function will get each hyperparameter
    setting and extend the runs done on this hyperparameter setting and saved
    in data1 by the runs of this hyperparameter setting and saved in data2.
    In short, this function extends the lists
    data1["experiment_data"][i]["runs"] by the lists
    data2["experiment_data"][i]["runs"] for all i. This is useful if
    multiple runs are done at different times, and the two data files need
    to be combined.

    Parameters
    ----------
    data1 : dict
        A data dictionary as generated by main.py
    data2 : dict
        A data dictionary as generated by main.py

    Raises
    ------
    KeyError
        If a hyperparameter setting exists in data2 but not in data1. This
        signals that the hyperparameter settings indices are most likely
        different, so the hyperparameter index i in data1 does not correspond
        to the same hyperparameter index in data2. In addition, all other
        functions expect the number of runs to be consistent for each
        hyperparameter setting, which would be violated in this case.
    """
    for hp_setting in data1["experiment_data"]:
        if hp_setting not in list(data2.keys()):
            # Ensure consistent hyperparam settings indices
            raise KeyError("hyperparameter settings are different " +
                           "between the two experiments")

        extra_runs = data2["experiment_data"][hp_setting]["runs"]
        data1["experiment_data"][hp_setting]["runs"].extend(extra_runs)


def get_returns(data, type_, ind, env_type="continuing"):
    """
    Gets the returns seen by an agent

    Gets the online or offline returns seen by an agent trained with
    hyperparameter settings index ind.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Whether to get the training or evaluation returns, one of 'train',
        'eval'
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    array_like
        The array of returns of the form (N, R, C) where N is the number of
        runs, R is the number of times a performance was measured, and C is the
        number of returns generated each time performance was measured
        (offline >= 1; online = 1). For the online setting, N is the number of
        runs, and R is the number of episodes and C = 1. For the offline
        setting, N is the number of runs, R is the number of times offline
        evaluation was performed, and C is the number of episodes run each
        time performance was evaluated offline.
    """
    if env_type == "episodic":
        data = runs.expand_episodes(data, ind, type_)

    returns = []
    if type_ == "eval":
        # Get the offline evaluation episode returns per run
        if data['experiment']['environment']['eval_episodes'] == 0:
            raise ValueError("cannot plot eval performance when " +
                             "experiment was run with eval_episodes = 0 in " +
                             "the configuration file")
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["eval_episode_rewards"])
        returns = np.stack(returns)

    elif type_ == "train":
        # Get the returns per episode per run
        for run in data["experiment_data"][ind]["runs"]:
            returns.append(run["train_episode_rewards"])
        returns = np.expand_dims(np.stack(returns), axis=-1)

    return returns


def get_mean_err(data, type_, ind, smooth_over, error,
                 env_type="continuing", keep_shape=False,
                 err_args={}):
    """
    Gets the timesteps, mean, and standard error to be plotted for
    a given hyperparameter settings index

    Note: This function assumes that each run has an equal number of episodes.
    This is true for continuing tasks. For episodic tasks, you will need to
    cutoff the episodes so all runs have the same number of episodes.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : int
        The hyperparameter settings index to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    error: function
        The error function to compute the error with
    env_type : str, optional
        The type of environment the data was generated on
    keep_shape : bool, optional
        Whether or not the smoothed data should discard or keep the first
        few data points before smooth_over.
    err_args : dict
        A dictionary of keyword arguments to pass to the error function

    Returns
    -------
    3-tuple of list(int), list(float), list(float)
        The timesteps, mean episodic returns, and standard errors of the
        episodic returns
    """
    timesteps = None  # So the linter doesn't have a temper tantrum

    # Determine the timesteps to plot at
    if type_ == "eval":
        timesteps = \
            data["experiment_data"][ind]["runs"][0]["timesteps_at_eval"]

    elif type_ == "train":
        timesteps_per_ep = \
            data["experiment_data"][ind]["runs"][0]["train_episode_steps"]
        timesteps = get_cumulative_timesteps(timesteps_per_ep)

    # Get the mean over all episodes per evaluation step (for online
    # returns, this axis will have length 1 so we squeeze it)
    returns = get_returns(data, type_, ind, env_type=env_type)
    returns = returns.mean(axis=-1)

    returns = smooth(returns, smooth_over, keep_shape=keep_shape)

    # Get the standard error of mean episodes per evaluation
    # step over all runs
    if error is not None:
        err = error(returns, **err_args)
    else:
        err = None

    # Get the mean over all runs
    mean = returns.mean(axis=0)

    # Return only the valid portion of timesteps. If smoothing and not
    # keeping the first data points, then the first smooth_over columns
    # will not have any data
    if not keep_shape:
        end = len(timesteps) - smooth_over + 1
        timesteps = timesteps[:end]

    return timesteps, mean, err


def stderr(matrix, axis=0):
    """
    Calculates the standard error along a specified axis

    Parameters
    ----------
    matrix : array_like
        The matrix to calculate standard error along the rows of
    axis : int, optional
        The axis to calculate the standard error along, by default 0

    Returns
    -------
    array_like
        The standard error of each row along the specified axis

    Raises
    ------
    np.AxisError
        If an invalid axis is passed in
    """
    if axis > len(matrix.shape) - 1:
        raise np.AxisError(f"""axis {axis} is out of bounds for array with
                           {len(matrix.shape) - 1} dimensions""")

    samples = matrix.shape[axis]
    return np.std(matrix, axis=axis) / np.sqrt(samples)


def smooth(matrix, smooth_over, keep_shape=False, axis=1):
    """
    Smooth the rows of returns

    Smooths the rows of returns by replacing the value at index i in a
    row of returns with the average of the next smooth_over elements,
    starting at element i.

    Parameters
    ----------
    matrix : array_like
        The array to smooth over
    smooth_over : int
        The number of elements to smooth over
    keep_shape : bool, optional
        Whether the smoothed array should have the same shape as
        as the input array, by default True. If True, then for the first
        few i < smooth_over columns of the input array, the element at
        position i is replaced with the average of all elements at
        positions j <= i.

    Returns
    -------
    array_like
        The smoothed over array
    """
    if smooth_over > 1:
        # Smooth each run separately
        kernel = np.ones(smooth_over) / smooth_over
        smoothed_matrix = _smooth(matrix, kernel, "valid", axis=axis)

        # Smooth the first few episodes
        if keep_shape:
            beginning_cols = []
            for i in range(1, smooth_over):
                # Calculate smoothing over the first i columns
                beginning_cols.append(matrix[:, :i].mean(axis=1))

            # Numpy will use each smoothed col as a row, so transpose
            beginning_cols = np.array(beginning_cols).transpose()
    else:
        return matrix

    if keep_shape:
        # Return the smoothed array
        return np.concatenate([beginning_cols, smoothed_matrix],
                              axis=1)
    else:
        return smoothed_matrix


def _smooth(matrix, kernel, mode="valid", axis=0):
    """
    Performs an axis-wise convolution of matrix with kernel

    Parameters
    ----------
    matrix : array_like
        The matrix to convolve
    kernel : array_like
        The kernel to convolve on each row of matrix
    mode : str, optional
         The mode of convolution, by default "valid". One of 'valid',
         'full', 'same'
    axis : int, optional
         The axis to perform the convolution along, by default 0

    Returns
    -------
    array_like
        The convolved array

    Raises
    ------
    ValueError
        If kernel is multi-dimensional
    """
    if len(kernel.shape) != 1:
        raise ValueError("kernel must be 1D")

    def convolve(mat):
        return np.convolve(mat, kernel, mode=mode)

    return np.apply_along_axis(convolve, axis=axis, arr=matrix)


def get_cumulative_timesteps(timesteps_per_episode):
    """
    Creates an array of cumulative timesteps.

    Creates an array of timesteps, where each timestep is the cumulative
    number of timesteps up until that point. This is needed for plotting the
    training data, where  the training timesteps are stored for each episode,
    and we need to plot on the x-axis the cumulative timesteps, not the
    timesteps per episode.

    Parameters
    ----------
    timesteps_per_episode : list
        A list where each element in the list denotes the amount of timesteps
        for the corresponding episode.

    Returns
    -------
    array_like
        An array where each element is the cumulative number of timesteps up
        until that point.
    """
    timesteps_per_episode = np.array(timesteps_per_episode)
    cumulative_timesteps = [timesteps_per_episode[:i].sum()
                            for i in range(timesteps_per_episode.shape[0])]

    return np.array(cumulative_timesteps)
