import pickle
import functools
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import sys
import seaborn as sns
import plot_utils as plot
import matplotlib as mpl
import experiment_utils as exp
import hypers


PLOT_REGION = True


params = {
      'axes.labelsize': 96,
      'axes.titlesize': 96,
      'legend.fontsize': 96,
      'xtick.labelsize': 96,
      'ytick.labelsize': 96,
  }
plt.rcParams.update(params)

plt.rc('text', usetex=False)  # You might want usetex=True to get Helvetica
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams.update({'font.size': 15})
plt.tick_params(top=False, right=False, labelsize=72)

sns.despine()

COLOURS = ["black", "red", "blue"]

HYPER = sys.argv[1]
PERFORMANCE_METRIC_TYPE = "train"


# Function definitions
def best_hyper(data, env_type, perf=PERFORMANCE_METRIC_TYPE):
    hypers = [np.finfo(np.float64).min] * len(data["experiment_data"])
    for hyper in data["experiment_data"]:
        hyper_data = []
        for run in data["experiment_data"][hyper]["runs"]:
            if env_type == CONTINUING:
                hyper_data.append(run[f"{perf}_episode_rewards"])
            elif env_type == EPISODIC:
                hyper_data.append(run[f"{perf}_episode_rewards"].mean())

        hyper_data = np.array(hyper_data)
        if perf == "train":
            if env_type == CONTINUING:
                hypers[hyper] = hyper_data.mean(axis=0).mean(axis=0)
            elif env_type == EPISODIC:
                hypers[hyper] = hyper_data.mean()
        else:
            ae_hypers[hyper] = hyper_data.mean(axis=-1).mean(axis=0).mean(
                axis=0)

    return np.argmax(hypers)


CONTINUING = "continuing"
EPISODIC = "episodic"
type_map = {
        "PendulumFixed-v0": EPISODIC,
        "Acrobot-v1": EPISODIC,
        "LunarLanderContinuous-v2": EPISODIC,
        "Bimodal1DEnv": CONTINUING,
        "Hopper-v2": EPISODIC,
        "PuddleWorld-v1": EPISODIC,
        "MountainCar-v0": EPISODIC,
        "MountainCarContinuous-v0": EPISODIC,
        "Pendulum-v0": CONTINUING,
        "Pendulum-v1": CONTINUING,
        "Walker2d": EPISODIC,
        "Swimmer-v2": EPISODIC
        }

env_jsons = [
    "config/environment/Final/PendulumContinuous-v0.json",
    "config/environment/Final/MountainCarContinuous-v1.json",
    "config/environment/Final/AcrobotContinuous-v1.json",
]
agent_jsons = [
    "config/agent/Final/GreedyAC.json",
    "config/agent/Final/SAC.json",
]

full_fig = plt.figure(figsize=(68, 32))
full_ax = full_fig.subplots(len(agent_jsons), len(env_jsons))

for j in range(len(env_jsons) * len(agent_jsons)):
    print(j)
    env_json = env_jsons[j % len(env_jsons)]
    # Load configuration files
    with open(env_json, "r") as infile:
        env_config = json.load(infile)
    if "gamma" not in env_config:
        env_config["gamma"] = -1

    agent_json = agent_jsons[j // len(env_jsons)]
    with open(agent_json, "r") as infile:
        agent_configs = [json.load(infile)]

    ENV = env_config["env_name"]
    ENV_TYPE = type_map[ENV]
    DATA_FILE = "data.pkl"

    DATA_FILES = []
    for config in agent_configs:
        agent = config["agent_name"]
        DATA_FILES.append(
            "./results/fixedEnt" +
            f"/{ENV}_{agent}results",
        )

    DATA = []
    for f in DATA_FILES:
        with open(os.path.join(f, DATA_FILE), "rb") as infile:
            DATA.append(pickle.load(infile))

    # Find best hypers
    BEST_IND = []
    for agent in DATA:
        best_hp = best_hyper(agent, env_type=ENV_TYPE)
        BEST_IND.append(best_hp)

    # Generate labels for plots
    labels = []
    for ag in DATA:
        labels.append([ag["experiment"]["agent"]["agent_name"]])
    colours = [["#003f5c"], ["#bc5090"], ["#ffa600"], ["#ff6361"], ["#58cfa1"]]

    HYPER_LIST = None
    YMIN = None
    YMAX = None

    means = []
    entropies = (0.001, 0.01, 0.1, 1.0, 10.0)
    for entropy in entropies:
        print("\t", entropy)
        # Plot the hyperparameter sensitivities
        LOW_RETURN = -1000
        HIGH_RETURN = 0
        # all_ax.set_ylim(LOW_RETURN, HIGH_RETURN)
        print("=== Plotting data hypers")
        for i, ag in tqdm(enumerate(DATA)):
            config = ag["experiment"]["agent"]
            print("\n", config["agent_name"])

            try:
                num_settings = hypers.total(config["parameters"])
                hps = config["parameters"][HYPER]
            except KeyError:
                num_settings = hypers.total(config["parameters"]["sweeps"])
                hps = config["parameters"]["sweeps"][HYPER]

            max_returns = [None] * len(hps)
            max_inds = [-1] * len(hps)

            for i in range(num_settings):
                try:
                    setting = hypers.sweeps(config["parameters"], i)[0]
                except KeyError:
                    setting = hypers.sweeps(config["parameters"]["sweeps"],
                                            i)[0]
                ind = hps.index(setting[HYPER])

                if entropy != setting["alpha"]:
                    continue
                if setting["betas"][0] != 0.9:
                    continue

                avg_return = []
                for run in ag["experiment_data"][i]["runs"]:
                    avg_return.append(run["train_episode_rewards"])

                if ENV_TYPE == EPISODIC:
                    avg_run_return = [np.mean(run) for run in avg_return]
                    avg_return = np.mean(avg_run_return)
                else:
                    avg_return = np.mean(avg_return)

                if max_returns[ind] is None or avg_return > max_returns[ind]:
                    max_inds[ind] = i
                    max_returns[ind] = avg_return

            returns = []
            print("\tIndices:", max_inds)
            for index in max_inds:
                alr = ag["experiment_data"][index]["agent_hyperparams"][
                    "actor_lr_scale"]
                clr = ag["experiment_data"][index]["agent_hyperparams"][
                    "critic_lr"]
                α = ag["experiment_data"][index]["agent_hyperparams"]["alpha"]

                if "betas" in ag["experiment_data"][index][
                   "agent_hyperparams"]:
                    β = ag["experiment_data"][index]["agent_hyperparams"][
                        "betas"]
                else:
                    β = None
                print(
                    f"\t\t{index}:\tactor_lr_scale={alr}" +
                    f"\tcritic_lr={clr}\tα={α}\tβ={β}",
                )

                index_returns = []
                for run in ag["experiment_data"][index]["runs"]:
                    if ENV_TYPE == EPISODIC:
                        index_returns.append(
                            run["train_episode_rewards"].mean())
                    else:
                        index_returns.append(run["train_episode_rewards"])
                returns.append(index_returns)

            returns = np.array(returns)
            if ENV_TYPE == EPISODIC:
                mean = returns.mean(axis=(-1))
                if YMIN is None or YMIN > np.min(mean):
                    YMIN = np.min(mean)
                if YMAX is None or YMAX < np.max(mean):
                    YMAX = np.max(mean)
                std_err = np.std(returns) / np.sqrt(returns.shape[1])
            else:
                mean = returns.mean(axis=(-1, -2))
                std_err = np.std(returns.mean(axis=1)) / np.sqrt(
                    returns.shape[1])
            print("\tMean returns:", mean)

            # fig = plt.figure()
            # ax = fig.add_subplot()

            # ax.set_ylabel("Average Return")
            # ax.set_xlabel("Hyperparameter Values")
            # # ax.set_ylim(LOW_RETURN, HIGH_RETURN)
            # # ax.set_xlim(0, 1.0)

            ag_name = ag["experiment"]["agent"]["agent_name"]
            # ax.set_title(ag_name + " " + HYPER + " " + ENV)

            # ax.plot(hps, mean)
            # ax.fill_between(hps, mean-std_err, mean+std_err, alpha=0.1)
            if HYPER_LIST is None:
                HYPER_LIST = hps

            means.append(mean)

    index = len(env_jsons) * (j // len(env_jsons)) + j % len(env_jsons)
    all_ax = full_ax.ravel()[index]
    # all_ax = full_ax

    means = np.array(means)
    print(means.shape)
    high = np.amax(means, axis=0)
    low = np.amin(means, axis=0)
    # all_ax.plot(hps, mean, label=ag_name+" entropy: "+str(entropy))
    c = COLOURS[index % len(COLOURS)]

    if PLOT_REGION:
        all_ax.fill_between(hps, low, high, alpha=1.0, color=c)
    else:
        for i in range(means.shape[0]):
            print(entropy)
            all_ax.plot(hps, means[i, :], label=f"entropy {entropies[i]}")
        all_ax.legend()

    # all_ax.fill_between(hps, mean-std_err, mean+std_err, alpha=0.1)

    all_ax.spines['top'].set_visible(False)
    all_ax.spines['right'].set_visible(False)
    all_ax.spines['bottom'].set_linewidth(2)
    all_ax.spines['left'].set_linewidth(2)

    # all_ax.get_xaxis().set_major_formatter(
    #       mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    full_fig.patch.set_facecolor('white')

    if HYPER == "actor_lr_scale":
        all_ax.set_xticks(HYPER_LIST)
        plt.xticks(rotation=45)

    if HYPER == "critic_lr":
        all_ax.set_xscale("log")
        plt.minorticks_off()
        YMIN = -1000
        YMAX = -100

    if "pendulum" in ENV.lower():
        all_ax.set_yticks([-1000, 0, 1000])
        all_ax.set_ylim(-1100, 1000)
    else:
        all_ax.set_yticks([-1000, -550, -100])
        all_ax.set_ylim(-1100, -100)

    all_ax.set_xticks(HYPER_LIST)
    labels = list(map(lambda x: str(int(np.log10(x))), HYPER_LIST))
    print("Labels:", labels)
    all_ax.set_xticklabels(labels)

    if j < len(env_jsons):
        if "mountaincar" in ENV.lower():
            env = "Mountain Car"
        elif "acrobot" in ENV.lower():
            env = "Acrobot"
        elif "pendulum" in ENV.lower():
            env = "Pendulum"
        all_ax.set_title(env)
    # all_ax.legend()
    AGENT = agent_configs[0]["agent_name"]

    if j == 0:
        all_ax.set_ylabel("Average Return", fontsize=96)
    elif j == len(env_jsons):
        all_ax.set_ylabel("Average Return", fontsize=96)
    elif (j+1) % len(env_jsons) == 0:
        all_ax.yaxis.set_label_position("right")
        if "cem" in AGENT.lower():
            agent = "GreedyAC"
        else:
            agent = AGENT.upper()
        all_ax.set_ylabel(agent, fontsize=96)

    if j > len(env_jsons) // len(agent_jsons) + 1:
        all_ax.set_xlabel("Critic Step-Size (10ˣ)", fontsize=96)

full_fig.tight_layout()
full_fig.savefig(f"/Users/Samuel/sensitivity.png", bbox_inches="tight")
full_fig.savefig(f"/Users/Samuel/sensitivity.svg", bbox_inches="tight")
full_fig.savefig(f"/Users/Samuel/sensitivity.pdf", bbox_inches="tight")
# full_ax.set_rasterized(True)  # for eps

exit(0)
