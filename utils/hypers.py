from functools import reduce
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
import pickle
from tqdm import tqdm
try:
    from utils.runs import expand_episodes
except ModuleNotFoundError:
    from runs import expand_episodes


TRAIN = "train"
EVAL = "eval"


def sweeps(parameters, index):
    """
    Gets the parameters for the hyperparameter sweep defined by the index.

    Each hyperparameter setting has a specific index number, and this function
    will get the appropriate parameters for the argument index. In addition,
    this the indices will wrap around, so if there are a total of 10 different
    hyperparameter settings, then the indices 0 and 10 will return the same
    hyperparameter settings. This is useful for performing loops.

    For example, if you had 10 hyperparameter settings and you wanted to do
    10 runs, the you could just call this for indices in range(0, 10*10). If
    you only wanted to do runs for hyperparameter setting i, then you would
    use indices in range(i, 10, 10*10)

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file
    index : int
        The index of the hyperparameters configuration to return

    Returns
    -------
    dict, int
        The dictionary of hyperparameters to use for the agent and the total
        number of combinations of hyperparameters (highest possible unique
        index)
    """
    # If the algorithm is a batch algorithm, ensure the batch size if less
    # than the replay buffer size
    if "batch_size" in parameters and "replay_capacity" in parameters:
        batches = np.array(parameters["batch_size"])
        replays = np.array(parameters["replay_capacity"])
        legal_settings = []

        # Calculate the legal combinations of batch sizes and replay capacities
        for batch in batches:
            legal = np.where(replays >= batch)[0]
            legal_settings.extend(list(zip([batch] *
                                           len(legal), replays[legal])))

        # Replace the configs batch/replay combos with the legal ones
        parameters["batch/replay"] = legal_settings
        replaced_hps = ["batch_size", "replay_capacity"]
    else:
        replaced_hps = []

    # Get the hyperparameters corresponding to the argument index
    out_params = {}
    accum = 1
    for key in parameters:
        if key in replaced_hps:
            # Ignore the HPs that have been sanitized and replaced by a new
            # set of HPs
            continue

        num = len(parameters[key])
        if key == "batch/replay":
            # Batch/replay must be treated differently
            batch_replay_combo = parameters[key][(index // accum) % num]
            out_params["batch_size"] = batch_replay_combo[0]
            out_params["replay_capacity"] = batch_replay_combo[1]
            accum *= num
            continue

        out_params[key] = parameters[key][(index // accum) % num]
        accum *= num

    return (out_params, accum)


def total(parameters):
    """
    Similar to sweeps but only returns the total number of
    hyperparameter combinations. This number is the total number of distinct
    hyperparameter settings. If this function returns k, then there are k
    distinct hyperparameter settings, and indices 0 and k refer to the same
    distinct hyperparameter setting.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file

    Returns
    -------
    int
        The number of distinct hyperparameter settings
    """
    return sweeps(parameters, 0)[1]


def satisfies(data, f):
    """
    Similar to hold_constant, except uses a function rather than a dictionary.
    Returns all hyperparameter settings that result in f evaluating to True.

    For each run, the hyperparameter dictionary for that run is inputted to f.
    If f returns True, then those hypers are kept.

    Parameters
    ----------
    data : dict
        The data dictionary generate from running an experiment
    f : f(dict) -> bool
        A function mapping hyperparameter settings (in a dictionary) to a
        boolean value

    Returns
    -------
    tuple of list[int], dict
        The list of hyperparameter settings satisfying the constraints
        defined by constant_hypers and a dictionary of new hyperparameters
        which satisfy these constraints
    """
    indices = []

    # Generate a new hyperparameter configuration based on the old
    # configuration
    new_hypers = deepcopy(data["experiment"]["agent"]["parameters"])

    # Clear the hyper configuration
    for key in new_hypers:
        if isinstance(new_hypers[key], list):
            new_hypers[key] = set()

    for index in data["experiment_data"]:
        hypers = data["experiment_data"][index]["agent_hyperparams"]
        if not f(hypers):
            continue

        # Track the hyper indices and the full hyper settings
        indices.append(index)
        for key in new_hypers:
            if key not in data["experiment_data"][index]["agent_hyperparams"]:
                # print(f"{key} not in agent hyperparameters, ignoring...")
                continue

            if isinstance(new_hypers[key], set):
                agent_val = data["experiment_data"][index][
                    "agent_hyperparams"][key]

                # Convert lists to a washable type
                if isinstance(agent_val, list):
                    agent_val = tuple(agent_val)

                new_hypers[key].add(agent_val)
            else:
                if key in new_hypers:
                    value = new_hypers[key]
                    raise IndexError("clobbering existing hyper " +
                                     f"{key} with value {value} with " +
                                     f"new value {agent_val}")
                new_hypers[key] = agent_val

    # Convert each set in new_hypers to a list
    for key in new_hypers:
        if isinstance(new_hypers[key], set):
            new_hypers[key] = sorted(list(new_hypers[key]))

    return indices, new_hypers


def index_of(hypers, equals):
    """
    Return the indices of agent hyperparameter settings that equals the single
    hyperparameter configuration equals.  The argument hypers is not modified.

    Parameters
    ----------
    hypers : dict[str]any
        A dictionary of agent hyperparameter settings, which may be a
        collection of hyperparameter configurations.
    equals : dict[ctr]any
        The hyperparameters that hypers should equal to. This should be a
        single hyperparameter configuration, and not a collection of such
        configurations.

    Returns
    -------
    list[ind]
        The list of indices in hypers which equals to equals
    """
    indices = []

    for i in range(total(hypers)):
        setting = sweeps(hypers, i)[0]

        if equal(setting, equals):
            indices.append(i)

    return indices


def equal(hyper1, hyper2):
    """
    Return whether two hyperparameter configurations are equal. These may be
    single configurations or collections of configurations.

    Parameters
    ----------
    hyper1 : dict[str]any
        One of the hyperparameter configurations to check equality for
    hyper2 : dict[str]any
        The other  hyperparameter configuration to check equality for

    Returns
    -------
    bool
        Whether the two hyperparameter settings are equal
    """
    newHyper1 = {}
    newHyper2 = {}
    for hyper in ("actor_lr_scale", "critic_lr"):
        newHyper1[hyper] = hyper1[hyper]
        newHyper2[hyper] = hyper2[hyper]

    hyper1 = newHyper1
    hyper2 = newHyper2

    # Ensure both hypers have the same keys
    if set(hyper1.keys()) != set(hyper2.keys()):
        return False

    equal = True
    for key in hyper1:
        value1 = hyper1[key]
        value2 = hyper2[key]
        if isinstance(value1, list):
            value1 = tuple(value1)
            value2 = tuple(value2)

        if value1 != value2:
            equal = False
            break

    return equal


def hold_constant(data, constant_hypers):
    """
    Returns the hyperparameter settings indices and hyperparameter values
    of the hyperparameter settings satisfying the constraints constant_hypers.

    Returns the hyperparameter settings indices in the data that
    satisfy the constraints as well as a new dictionary of hypers which satisfy
    the constraints. The indices returned are the hyper indices of the original
    data and not the indices into the new hyperparameter configuration
    returned.

    Parameters
    ----------
    data: dict
        The data dictionary generated from an experiment

    constant_hypers: dict[string]any
        A dictionary mapping hyperparameters to a value that they should be
        equal to.

    Returns
    -------
    tuple of list[int], dict
        The list of hyperparameter settings satisfying the constraints
        defined by constant_hypers and a dictionary of new hyperparameters
        which satisfy these constraints

    Example
    -------
    >>> data = ...
    >>> contraints = {"stepsize": 0.8}
    >>> hold_constant(data, constraints)
    (
        [0, 1, 6, 7],
        {
            "stepsize": [0.8],
            "decay":    [0.0, 0.5],
            "epsilon":  [0.0, 0.1],
        }
    )
    """
    indices = []

    # Generate a new hyperparameter configuration based on the old
    # configuration
    new_hypers = deepcopy(data["experiment"]["agent"]["parameters"])
    # Clear the hyper configuration
    for key in new_hypers:
        if isinstance(new_hypers[key], list):
            new_hypers[key] = set()

    # Go through each hyperparameter index, checking if it satisfies the
    # constraints
    for index in data["experiment_data"]:
        # Assume we hyperparameter satisfies the constraints
        constraint_satisfied = True

        # Check to see if the agent hyperparameter satisfies the constraints
        for hyper in constant_hypers:
            constant_val = constant_hypers[hyper]

            # Ensure the constrained hyper exists in the data
            if hyper not in data["experiment_data"][index][
               "agent_hyperparams"]:
                raise IndexError(f"no such hyper {hyper} in agent hypers")

            agent_val = data["experiment_data"][index]["agent_hyperparams"][
                hyper]

            if agent_val != constant_val:
                # Hyperparameter does not satisfy the constraints
                constraint_satisfied = False
                break

        # If the constraint is satisfied, then we will store the hypers
        if constraint_satisfied:
            indices.append(index)

            # Add the hypers to the configuration
            for key in new_hypers:
                if key == "batch/replay":
                    continue
                if isinstance(new_hypers[key], set):
                    agent_val = data["experiment_data"][index][
                        "agent_hyperparams"][key]

                    if isinstance(agent_val, list):
                        agent_val = tuple(agent_val)

                    new_hypers[key].add(agent_val)
                else:
                    if key in new_hypers:
                        value = new_hypers[key]
                        raise IndexError("clobbering existing hyper " +
                                         f"{key} with value {value} with " +
                                         f"new value {agent_val}")
                    new_hypers[key] = agent_val

    # Convert each set in new_hypers to a list
    for key in new_hypers:
        if isinstance(new_hypers[key], set):
            new_hypers[key] = sorted(list(new_hypers[key]))

    return indices, new_hypers


def _combine_two(data1, data2, config):
    """
    Combine two data dictionaries into one, with hypers renumbered to satisfy
    the configuration config

    Parameters
    ----------
    data1 : dict
        The first data dictionary
    data2 : dict
        The second data dictionary
    config : dict
        The hyperparameter configuration

    Returns
    -------
    dict
        The combined data dictionary
    """
    agent1_name = data1["experiment"]["agent"]["agent_name"].lower()
    agent2_name = data2["experiment"]["agent"]["agent_name"].lower()
    config_agent_name = config["agent_name"].lower()
    if agent1_name != agent2_name or config_agent_name != agent1_name:
        raise ValueError("all data should be generate by the same agent " +
                         f"but got agents {agent1_name}, {agent2_name}, " +
                         f"and {config_agent_name}")

    # Renumber of the configuration file does not match that with which the
    # experiment was run
    if data1["experiment"]["agent"]["parameters"] != config["parameters"]:
        data1 = renumber(data1, config["parameters"])
    if data2["experiment"]["agent"]["parameters"] != config["parameters"]:
        data2 = renumber(data2, config["parameters"])

    new_data = {}
    new_data["experiment"] = data1["experiment"]
    new_data["experiment_data"] = {}

    for hyper in data1["experiment_data"]:
        new_data["experiment_data"][hyper] = data1["experiment_data"][hyper]

    for hyper in data2["experiment_data"]:
        # Before we extend the data, ensure we do not overwrite any runs
        if hyper in new_data["experiment_data"]:
            # Get a map of run number -> random seed from the already combined
            # data
            seeds = []
            for run in new_data["experiment_data"][hyper]["runs"]:
                seeds.append(run["random_seed"])

            for run in data2["experiment_data"][hyper]["runs"]:
                seed = run["random_seed"]

                # Don't add a run if it already exists in the combined data. A
                # run exists in the combined data if its seed has been used.
                if seed in seeds:
                    continue
                else:
                    # Run does not exist in the data
                    new_data["experiment_data"][hyper]["runs"].append(run)

        else:
            new_data["experiment_data"][hyper] = \
                data2["experiment_data"][hyper]

    return new_data


def combine(config, *data):
    """
    Combines a number of data dictionaries, renumbering the hyper settings to
    satisfy config.

    Parameters
    ----------
    config : dict
        The hyperparameter configuration
    *data : iterable of dict
        The data dictionaries to combine

    Returns
    -------
    dict
        The combined data dictionary
    """
    config_agent_name = config["agent_name"].lower()
    for d in data:
        agent_name = d["experiment"]["agent"]["agent_name"].lower()
        if agent_name != config_agent_name:
            raise ValueError("all data should be generate by the same agent " +
                             f"but got agents {agent_name} and " +
                             f"{config_agent_name}")

    return reduce(lambda x, y: _combine_two(x, y, config), data)


def renumber(data, hypers):
    """
    Renumbers the hyperparameters in data to reflect the hyperparameter map
    hypers. If any hyperparameter settings exist in data that do not exist in
    hypers, then those data are discarded.

    Note that each hyperparameter listed in hypers must also be listed in data
    and vice versa, but the specific hyperparameter values need not be the
    same. For example if "decay" ∈ data[hypers], then it also must be in hypers
    and vice versa. If 0.9 ∈ data[hypers][decay], then it need *not* be in
    hypers[decay].

    This function does not mutate the input data, but rather returns a copy of
    the input data, appropriately mutated.

    Parameters
    ----------
    data : dict
        The data dictionary generated from running the experiment
    hypers : dict
        The new dictionary of hyperparameter values

    Returns
    -------
    dict
        The modified data dictionary

    Examples
    --------
    >>> data = ...
    >>> contraints = {"stepsize": 0.8}
    >>> new_hypers = hold_constant(data, constraints)[1]
    >>> new_data = renumber(data, new_hypers)
    """
    if len(hypers) == 0:
        return data

    if hypers == data["experiment"]["agent"]["parameters"]:
        return data

    # Ensure each hyperparameter is in both hypers and data; hypers need not
    # list every hyperparameter *value* that is listed in data, but it needs to
    # have the same hyperparameters. E.g. if "decay" exists in data then it
    # should also exist in hypers, but if 0.9 ∈ data[hypers][decay], this value
    # need not exist in hypers.
    for key in data["experiment"]["agent"]["parameters"]:
        if key not in hypers and key != "batch/replay":
            raise ValueError("data and hypers should have all the same " +
                             f"hyperparameters but {key} ∈ data but ∉ hypers")

    # Ensure each hyperparameter listed in hypers is also listed in data. If it
    # isn't then it isn't clear which value of this hyperparamter the data in
    # data should map to. E.g. if "decay" = [0.1, 0.2] ∈ hypers but ∉ data,
    # which value should we set for the data in data when renumbering? 0.1 or
    # 0.2?
    for key in hypers:
        if key not in data["experiment"]["agent"]["parameters"]:
            raise ValueError("data and hypers should have all the same " +
                             f"hyperparameters but {key} ∈ hypers but ∉ data")

    new_data = {}
    new_data["experiment"] = data["experiment"]
    new_data["experiment"]["agent"]["parameters"] = hypers
    new_data["experiment_data"] = {}

    total_hypers = total(hypers)

    for i in range(total_hypers):
        setting = sweeps(hypers, i)[0]

        for j in data["experiment_data"]:
            agent_hypers = data["experiment_data"][j]["agent_hyperparams"]
            setting_in_data = True

            # For each hyperparameter value in setting, ensure that the
            # corresponding agent hyperparameter is equal. If not, ignore that
            # hyperparameter setting.
            for key in setting:
                # If the hyper setting is iterable, then check each value in
                # the iterable to ensure it is equal to the corresponding
                # value in the agent hyperparameters
                if isinstance(setting[key], Iterable):
                    if len(setting[key]) != len(agent_hypers[key]):
                        setting_in_data = False
                        break
                    for k in range(len(setting[key])):
                        if setting[key][k] != agent_hypers[key][k]:
                            setting_in_data = False
                            break

                # Non-iterable data
                elif setting[key] != agent_hypers[key]:
                    setting_in_data = False
                    break

            if setting_in_data:
                new_data["experiment_data"][i] = data["experiment_data"][j]

    return new_data


def get_performance(data, hyper, type_=TRAIN, repeat=True):
    """
    Returns the data for each run of key, optionally adjusting the runs'
    data so that each run has the same number of data points. This is
    accomplished by repeating each episode's performance by the number of
    timesteps the episode took to complete

    Parameters
    ----------
    data : dict
        The data dictionary
    hyper : int
        The hyperparameter index to get the run data of
    repeat : bool
        Whether or not to repeat the runs data

    Returns
    -------
    np.array
        The array of performance data
    """
    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"unknown type {type_}")

    key = type_ + "_episode_rewards"

    if repeat:
        data = expand_episodes(data, hyper, type_)

    run_data = []
    for run in data["experiment_data"][hyper]["runs"]:
        run_data.append(run[key])

    return np.array(run_data)


def best_from_files(files, num_hypers=None, perf=TRAIN,
                    scale_by_episode_length=False, to=-1):
    """
    This function is like `best`, but looks through a list of files rather than
    a single data dictionary.

    If `num_hypers` is `None`, then finds total number of hyper settings from
    the data files. Otherwise, assumes `num_hypers` hyper settings exist in the
    data.
    """
    # Get the hyperparameter indices in files
    if num_hypers is None:
        hyper_inds = set()
        print("Finding total number of hyper settings")

        for file in tqdm(files):
            with open(file, "rb") as infile:
                d = pickle.load(infile)
            hyper_inds.update(d["experiment_data"].keys())
        num_hypers = len(hyper_inds)

    hypers = [np.finfo(np.float64).min] * num_hypers

    print("Finding best hyper setting")
    hyper_to_files = {}
    for file in tqdm(files):
        with open(file, "rb") as infile:
            data = pickle.load(infile)

        for hyper in data["experiment_data"]:
            hyper_data = []

            # Store a dictionary of hyper indices to files that contain them
            if hyper not in hyper_to_files:
                hyper_to_files[hyper] = []
            else:
                hyper_to_files[hyper].append(file)

            for run in data["experiment_data"][hyper]["runs"]:
                if to <= 0:
                    # Tune over all timesteps
                    returns = np.array(run[f"{perf}_episode_rewards"])
                    scale = np.array(run[f"{perf}_episode_steps"])
                else:
                    # Tune only to timestep determined by parameter to
                    cum_steps = np.cumsum(run[f"{perf}_episode_steps"])
                    returns = np.array(run[f"{perf}_episode_rewards"])
                    scale = np.array(run[f"{perf}_episode_steps"])

                    # If the total number of steps we ran the experiment for is
                    # more than the number of steps we want to tune to, then
                    # truncate the trailing data and tune only to the
                    # appropriate timestep
                    if cum_steps[-1] > to:
                        last_step = np.argmax(cum_steps > to)
                        returns = returns[:last_step+1]
                        scale = scale[:last_step + 1]

                        if scale_by_episode_length:
                            # Rescale the last episode such that we only
                            # consider the timesteps up to the argument to, and
                            # not beyond that
                            if len(scale) > 1:
                                last_ep_scale = (to - scale[-2])
                            else:
                                last_ep_scale = to
                            scale[-1] = last_ep_scale

                if scale_by_episode_length:
                    pass
                    returns *= scale
                hyper_data.append(returns.mean())

            hyper_data = np.array(hyper_data)
            hypers[hyper] = hyper_data.mean()

        del data

    argmax = np.argmax(hypers)
    return argmax, hypers[argmax], hyper_to_files[argmax]


def best(data, perf=TRAIN, scale_by_episode_length=False, to=-1):
    """
    Returns the hyperparameter index of the hyper setting which resulted in the
    highest AUC of the learning curve. AUC is calculated by computing the AUC
    for each run, then taking the average over all runs.

    Parameters
    ----------
    data : dict
        The data dictionary
    perf : str
        The type of performance to evaluate, train or eval
    scale_by_episode_length : bool
        Whether or not each return should be scaled by the length of the
        episode. This is useful for episodic tasks, but has no effect in
        continuing tasks as long as the value of the parameter to does not
        result in a episode being truncated.
    to : int
        Only tune to this timestep. If <= 0, then all timesteps are considered.
        If `scale_by_episode_length == True`, then tune based on all timesteps
        up until this timestep. If `scale_by_episode_length == False`, then
        tune based on all episodes until the episode which contains this
        timestep (inclusive).

    Returns
    -------
    np.array[int], np.float32
        The hyper settings that resulted in the maximum return as well as the
        maximum return
    """
    max_hyper = int(np.max(list(data["experiment_data"].keys())))
    hypers = [np.finfo(np.float64).min] * (max_hyper + 1)

    for hyper in data["experiment_data"]:
        hyper_data = []
        for run in data["experiment_data"][hyper]["runs"]:
            if to <= 0:
                # Tune over all timesteps
                returns = np.array(run[f"{perf}_episode_rewards"])
                scale = np.array(run[f"{perf}_episode_steps"])
            else:
                # Tune only to timestep determined by parameter to
                cum_steps = np.cumsum(run[f"{perf}_episode_steps"])
                returns = np.array(run[f"{perf}_episode_rewards"])
                scale = np.array(run[f"{perf}_episode_steps"])

                # If the total number of steps we ran the experiment for is
                # more than the number of steps we want to tune to, then
                # truncate the trailing data and tune only to the appropriate
                # timestep
                if cum_steps[-1] > to:
                    last_step = np.argmax(cum_steps > to)
                    returns = returns[:last_step+1]
                    scale = scale[:last_step + 1]

                    if scale_by_episode_length:
                        # Rescale the last episode such that we only
                        # consider the timesteps up to the argument to, and
                        # not beyond that
                        if len(scale) > 1:
                            last_ep_scale = (to - scale[-2])
                        else:
                            last_ep_scale = to
                        scale[-1] = last_ep_scale

            if scale_by_episode_length:
                returns *= scale
            hyper_data.append(returns.mean())

        hyper_data = np.array(hyper_data)
        hypers[hyper] = hyper_data.mean()

    return np.argmax(hypers), np.max(hypers)


def get(data, ind):
    """
    Gets the hyperparameters for hyperparameter settings index ind

    data : dict
        The Python data dictionary generated from running main.py
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index

    Returns
    -------
    dict
        The dictionary of hyperparameters
    """
    return data["experiment_data"][ind]["agent_hyperparams"]


def which(data, hypers, equal_keys=False):
    """
    Get the hyperparameter index at which all agent hyperparameters are
    equal to those specified by hypers.

    Parameters
    ----------
    data : dict
        The data dictionary that resulted from running an experiment
    hypers : dict[string]any
        A dictionary of hyperparameters to the values that those
        hyperparameters should take on
    equal_keys : bool, optional
        Whether or not all keys must be shared between the sets of agent
        hyperparameters and the argument hypers. By default False.

    Returns
    -------
    int, None
        The hyperparameter index at which the agent had hyperparameters equal
        to those specified in hypers.

    Examples
    --------
    >>> data = ... # Some data from an experiment
    >>> hypers = {"critic_lr": 0.01, "actor_lr": 1.0}
    >>> ind = which(data, hypers)
    >>> print(ind in data["experiment_data"])
        True
    """
    for ind in data["experiment_data"]:
        is_equal = True
        agent_hypers = data["experiment_data"][ind]["agent_hyperparams"]

        # Ensure that all keys in each dictionary are equal
        if equal_keys and set(agent_hypers.keys()) != set(hypers.keys()):
            continue

        # For the current set of agent hyperparameters (index ind), check to
        # see if all hyperparameters used by the agent are equal to those
        # specified by hypers. If not, then break and check the next set of
        # agent hyperparameters.
        for h in hypers:
            if h in agent_hypers and hypers[h] != agent_hypers[h]:
                is_equal = False
                break

        if is_equal:
            return ind

    # No agent hyperparameters were found that coincided with the argument
    # hypers
    return None
