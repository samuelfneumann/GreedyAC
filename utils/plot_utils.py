# Import modules
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker, gridspec
import experiment_utils as exp
import numpy as np
from scipy import ndimage
from scipy import stats as st
import seaborn as sns
from collections.abc import Iterable
import pickle
import matplotlib as mpl
import hypers
import warnings
import runs

TRAIN = "train"
EVAL = "eval"


# Set up plots
params = {
      'axes.labelsize': 48,
      'axes.titlesize': 36,
      'legend.fontsize': 16,
      'xtick.labelsize': 48,
      'ytick.labelsize': 48
}
plt.rcParams.update(params)

plt.rc('text', usetex=False)  # You might want usetex=True to get DejaVu Sans
plt.rc('font', **{'family': 'sans-serif', 'serif': ['DejaVu Sans']})
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({'font.size': 15})
plt.tick_params(top=False, right=False, labelsize=20)

mpl.rcParams["svg.fonttype"] = "none"


# Constants
EPISODIC = "episodic"
CONTINUING = "continuing"


# Colours
CMAP = "tab10"
DEFAULT_COLOURS = list(sns.color_palette(CMAP, 6).as_hex())
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=sns.color_palette(CMAP))
OFFSET = 0  # The offset to start in DEFAULT_COLOURS


def mean_with_err(data, type_, ind, smooth_over, names, fig=None, ax=None,
                  figsize=(12, 6), xlim=None, ylim=None, alpha=0.1,
                  colours=None, linestyles=None, markers=None,
                  env_type="continuing", keep_shape=False, xlabel="",
                  ylabel="", skip=-1, errfn=exp.stderr, linewidth=1,
                  markersize=1):
    """
    Plots the average training or evaluation return over all runs with standard
    error.

    Given a list of data dictionaries of the form returned by main.py, this
    function will plot each episodic return for the list of hyperparameter
    settings ind each data dictionary. The ind argument is a list, where each
    element is a list of hyperparameter settings to plot for the data
    dictionary at the same index as this list. For example, if ind[i] = [1, 2],
    then plots will be generated for the data dictionary at location i
    in the data argument for hyperparameter settings ind[i] = [1, 2].
    The smooth_over argument tells how many previous data points to smooth
    over

    Parameters
    ----------
    data : list of dict
        The Python data dictionaries generated from running main.py for the
        agents
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of iter of int
        The list of lists of hyperparameter settings indices to plot for
        each agent. For example [[1, 2], [3, 4]] means that the first agent
        plots will use hyperparameter settings indices 1 and 2, while the
        second will use 3 and 4.
    smooth_over : list of int
        The number of previous data points to smooth over for the agent's
        plot for each data dictionary. Note that this is *not* the number of
        timesteps to smooth over, but rather the number of data points to
        smooth over. For example, if you save the return every 1,000
        timesteps, then setting this value to 15 will smooth over the last
        15 readings, or 15,000 timesteps. For example, [1, 2] will mean that
        the plots using the first data dictionary will smooth over the past 1
        data points, while the second will smooth over the passed 2 data
        points for each hyperparameter setting.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    names : list of str
        The name of the agents, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of list of str
        The colours to use for each hyperparameter settings plot for each data
        dictionary
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot
    """
    # Set the colours to be default if not set
    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        linestyles = _get_default_styles(ind)
    if markers is None:
        markers = _get_default_markers(ind)

    # Set up figure
    title = f"Average {type_.title()} Return per Run with Standard Error"
    fig, ax = _setup_fig(fig, ax, figsize, xlim=xlim, ylim=ylim, xlabel=xlabel,
                         ylabel=ylabel, title=title)

    # Track the total timesteps per hyperparam setting over all episodes and
    # the cumulative timesteps per episode per data dictionary (timesteps
    # should be consistent between all hp settings in a single data dict)
    total_timesteps = []
    cumulative_timesteps = []

    for i in range(len(data)):
        if type_ == "train":
            cumulative_timesteps.append(exp.get_cumulative_timesteps(data[i]
                                        ["experiment_data"][ind[i][0]]["runs"]
                                        [0]["train_episode_steps"]))
        elif type_ == "eval":
            cumulative_timesteps.append(data[i]["experiment_data"][ind[i][0]]
                                        ["runs"][0]["timesteps_at_eval"])
        else:
            raise ValueError("type_ must be one of 'train', 'eval'")
        total_timesteps.append(cumulative_timesteps[-1][-1])

    # Find the minimum of total trained-for timesteps. Each plot will only
    # be plotted on the x-axis until this value
    min_timesteps = min(total_timesteps)

    # For each data dictionary, find the minimum index where the timestep at
    # that index is >=  minimum timestep
    ind_ge_min_timesteps = []
    for cumulative_timesteps_per_data in cumulative_timesteps:
        final_ind = np.where(cumulative_timesteps_per_data >=
                             min_timesteps)[0][0]
        # Since indexing will stop right before the minimum, increment it
        ind_ge_min_timesteps.append(final_ind + 1)

    # Plot all data for all HP settings, only up until the minimum index
    plot_fn = _plot_mean_with_err_continuing if env_type == "continuing" \
        else _plot_mean_with_err_episodic
    for i in range(len(data)):
        fig, ax = \
            plot_fn(data=data[i], type_=type_,
                    ind=ind[i], smooth_over=smooth_over[i], name=names[i],
                    fig=fig, ax=ax, figsize=figsize, xlim=xlim, ylim=ylim,
                    last_ind=ind_ge_min_timesteps[i], alpha=alpha,
                    colours=colours[i], keep_shape=keep_shape, skip=skip,
                    linestyles=linestyles[i], markers=markers[i],
                    errfn=errfn, linewidth=linewidth, markersize=markersize)

    return fig, ax


def _plot_mean_with_err_continuing(data, type_, ind, smooth_over, fig=None,
                                   ax=None, figsize=(12, 6), xlim=None,
                                   ylim=None, xlabel=None, ylabel=None,
                                   name="", last_ind=-1,
                                   timestep_multiply=None, alpha=0.1,
                                   colours=None, linestyles=None, markers=None,
                                   keep_shape=False, skip=-1, errfn=exp.stderr,
                                   linewidth=1, markersize=1):
    """
    Plots the average training or evaluation return over all runs for a single
    data dictionary on a continuing environment. Standard error
    is plotted as shaded regions.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure
        The figure to plot on, by default None. If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    last_ind : int, optional
        The index of the last element to plot in the returns list,
        by default -1. This is useful if you want to plot many things on the
        same axis, but all of which have a different number of elements. This
        way, we can plot the first last_ind elements of each returns for each
        agent.
    timestep_multiply : array_like of float, optional
        A value to multiply each timstep by, by default None. This is useful if
        your agent does multiple updates per timestep and you want to plot
        performance vs. number of updates.
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if skip == 0:
        skip = 1

    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")
    if linestyles is not None and len(linestyles) != len(ind):
        raise ValueError("must have one linestyle for each hyperparameter " +
                         "setting")
    if markers is not None and len(markers) != len(ind):
        raise ValueError("must have one marker for each hyperparameter " +
                         "setting")

    if timestep_multiply is None:
        timestep_multiply = [1] * len(ind)

    if ax is not None and fig is None:
        raise ValueError("must pass figure when passing axis")

    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        colours = _get_default_styles(ind)
    if markers is None:
        colours = _get_default_markers(ind)

    # Set up figure
    if ax is None and fig is None:
        title = f"Average {type_.title()} Return per Run with Standard Error"
        fig, ax = _setup_fig(fig, ax, figsize, xlim=xlim, ylim=ylim,
                             xlabel=xlabel, ylabel=ylabel, title=title)

    episode_length = data["experiment"]["environment"]["steps_per_episode"]

    # Plot with the standard error
    for i in range(len(ind)):
        timesteps, mean, std = exp.get_mean_err(data, type_, ind[i],
                                                smooth_over,
                                                # exp.t_ci,
                                                errfn,
                                                keep_shape=keep_shape)
        timesteps = np.array(timesteps[:last_ind]) * timestep_multiply[i]
        timesteps = timesteps[::skip]
        mean = mean[::skip]
        std = std[::skip]

        # Plot based on colours
        label = f"{name}"
        if colours is not None and linestyles is not None:
            _plot_shaded(ax, timesteps, mean, std, colours[i], label, alpha,
                         linestyle=linestyles[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize)
        elif colours is not None:
            _plot_shaded(ax, timesteps, mean, std, colours[i], label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize)
        elif linestyles is not None:
            _plot_shaded(ax, timesteps, mean, std, None, label, alpha,
                         linestyle=linestyle[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize)
        else:
            _plot_shaded(ax, timesteps, mean, std, None, label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize)

    ax.legend()

    fig.show()
    return fig, ax


def _plot_mean_with_err_episodic(data, type_, ind, smooth_over, fig=None,
                                 ax=None, figsize=(12, 6), name="",
                                 last_ind=-1, xlabel="Timesteps",
                                 ylabel="Average Return", xlim=None, ylim=None,
                                 alpha=0.1, colours=None, keep_shape=False,
                                 skip=-1, linestyles=None, markers=None,
                                 errfn=exp.stderr, linewidth=1, markersize=1):
    """
    Plots the average training or evaluation return over all runs for a
    single data dictionary on an episodic environment. Plots shaded retions
    as standard error.

    Parameters
    ----------
    data : dict
        The Python data dictionary generated from running main.py
    type_ : str
        Which type of data to plot, one of "eval" or "train"
    ind : iter of int
        The list of hyperparameter settings indices to plot
    smooth_over : int
        The number of previous data points to smooth over. Note that this
        is *not* the number of timesteps to smooth over, but rather the number
        of data points to smooth over. For example, if you save the return
        every 1,000 timesteps, then setting this value to 15 will smooth
        over the last 15 readings, or 15,000 timesteps.
    fig : plt.figure The figure to plot on, by default None.
    If None, creates a new figure
    ax : plt.Axes
        The axis to plot on, by default None, If None, creates a new axis
    figsize : tuple(int, int)
        The size of the figure to plot
    name : str, optional
        The name of the agent, used for the legend
    xlim : float, optional
        The x limit for the plot, by default None
    ylim : float, optional
        The y limit for the plot, by default None
    alpha : float, optional
        The alpha channel for the plot, by default 0.1
    colours : list of str
        The colours to use for each plot of each hyperparameter setting
    env_type : str, optional
        The type of environment, one of 'continuing', 'episodic'. By default
        'continuing'

    Returns
    -------
    plt.figure, plt.Axes
        The figure and axes of the plot

    Raises
    ------
    ValueError
        When an axis is passed but no figure is passed
        When an appropriate number of colours is not specified to cover all
        hyperparameter settings
    """
    if colours is not None and len(colours) != len(ind):
        raise ValueError("must have one colour for each hyperparameter " +
                         "setting")
    if linestyles is not None and len(colours) != len(ind):
        raise ValueError("must have one linestyle for each hyperparameter " +
                         "setting")
    if markers is not None and len(markers) != len(ind):
        raise ValueError("must have one marker for each hyperparameter " +
                         "setting")

    if ax is not None and fig is None:
        raise ValueError("must pass figure when passing axis")

    if colours is None:
        colours = _get_default_colours(ind)
    if linestyles is None:
        linestyles = _get_default_styles(ind)
    if markers is None:
        markers = _get_default_markers(ind)

    # Set up figure
    if ax is None and fig is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Plot with the standard error
    for i in range(len(ind)):
        # data = exp.reduce_episodes(data, ind[i], type_=type_)
        if type_ == "train":
            data = runs.expand_episodes(data, ind[i], type_=type_, skip=skip)

        # data has consistent # of episodes, so treat as env_type="continuing"
        _, mean, std = exp.get_mean_err(data, type_, ind[i], smooth_over,
                                        errfn,
                                        keep_shape=keep_shape)
        episodes = np.arange(mean.shape[0])

        # Plot based on colours
        label = f"{name}"
        if colours is not None and linestyles is not None:
            _plot_shaded(ax, episodes, mean, std, colours[i], label, alpha,
                         linestyle=linestyles[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize)
        elif colours is not None:
            _plot_shaded(ax, episodes, mean, std, colours[i], label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize)
        elif linestyles is not None:
            _plot_shaded(ax, episodes, mean, std, None, label, alpha,
                         linestyle=linestyles[i], marker=markers[i],
                         linewidth=linewidth, markersize=markersize)
        else:
            _plot_shaded(ax, episodes, mean, std, None, label, alpha,
                         marker=markers[i], linewidth=linewidth,
                         markersize=markersize)

    ax.legend()
    ax.set_title(f"Average {type_.title()} Return per Run with Standard Error")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # fig.show()
    return fig, ax


def _get_default_markers(iter_):
    markers = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        return ["" for i in paths]

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        markers.append(_get_default_markers(iter_[i]))

    return markers


def _get_default_styles(iter_):
    styles = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        return ["-" for i in paths]

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        styles.append(_get_default_styles(iter_[i]))

    return styles


def _get_default_colours(iter_):
    """
    Recursively turns elements of an Iterable into strings representing
    colours.

    This function will turn each element of an Iterable into strings that
    represent colours, recursively. If the elements of an Iterable are
    also Iterable, then this function will recursively descend all the way
    through every Iterable until it finds an Iterable with non-Iterable
    elements. These elements will be replaced by strings that represent
    colours. In effect, this function keeps the data structure, but replaces
    non-Iterable elements by strings representing colours. Note that this
    funcion assumes that all elements of an Iterable are of the same type,
    and so it only checks if the first element of an Iterable object is
    Iterable or not to stop the recursion.

    Parameters
    ----------
    iter_ : collections.Iterable
        The top-level Iterable object to turn into an Iterable of strings of
        colours, recursively.

    Returns
    -------
    list of list of ... of strings
        A data structure that has the same architecture as the input Iterable
        but with all non-Iterable elements replaced by strings.
    """
    colours = []

    # Calculate the number of lists at the current level to go through
    paths = range(len(iter_))

    # Return a list of colours if the elements of the list are not lists
    if not isinstance(iter_[0], Iterable):
        global OFFSET
        col = [DEFAULT_COLOURS[(OFFSET + i) % len(DEFAULT_COLOURS)]
               for i in paths]
        OFFSET += len(paths)
        return col

    # For each list at the current level, get the colours corresponding to
    # this level
    for i in paths:
        colours.append(_get_default_colours(iter_[i]))

    return colours


def _plot_shaded(ax, x, y, region, colour, label, alpha, linestyle="-",
                 marker="", linewidth=1, markersize=1):
    """
    Plots a curve with a shaded region.

    Parameters
    ----------
    ax : plt.Axes
        The axis to plot on
    x : Iterable
        The points on the x-axis
    y : Iterable
        The points on the y-axis
    region : list or array_like
        The region to shade about the y points. The shaded region will be
        y +/- region. If region is a list or 1D np.ndarray, then the region
        is used both for the + and - portions. If region is a 2D np.ndarray,
        then the first row will be used as the lower bound (-) and the
        second row will be used for the upper bound (+). That is, the region
        between (lower_bound, upper_bound) will be shaded, and there will be
        no subtraction/adding of the y-values.
    colour : str
        The colour to plot with
    label : str
        The label to use for the plot
    alpha : float
        The alpha value for the shaded region
    """
    if colour is None:
        colour = DEFAULT_COLOURS[0]

    ax.plot(x, y, color=colour, label=label, linestyle=linestyle,
            marker=marker, linewidth=linewidth, markersize=markersize,
            markevery=3)
    if type(region) == list:
        ax.fill_between(x, y-region, y+region, alpha=alpha, color=colour)
    elif type(region) == np.ndarray and len(region.shape) == 1:
        ax.fill_between(x, y-region, y+region, alpha=alpha, color=colour)
    elif type(region) == np.ndarray and len(region.shape) == 2:
        ax.fill_between(x, region[0, :], region[1, :], alpha=alpha,
                        color=colour)


def _setup_fig(fig, ax, figsize=None, title=None, xlim=None, ylim=None,
               xlabel=None, ylabel=None, xscale=None, yscale=None, xbase=None,
               ybase=None):
    if fig is None:
        if ax is not None:
            raise ValueError("Must specify figure when axis given")
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()

    if ax is None:
        ax = fig.add_subplot()

    if title is not None:
        ax.set_title(title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xscale is not None:
        if xbase is not None:
            ax.set_xscale(xscale, base=xbase)
        else:
            ax.set_xscale(xscale)

    if yscale is not None:
        if ybase is not None:
            ax.set_yscale(yscale, base=ybase)
        else:
            ax.set_yscale(yscale)

    return fig, ax
