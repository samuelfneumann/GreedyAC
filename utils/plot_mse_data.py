# Plot mean with standard error
# Input to script is the data files to consider

from pprint import pprint
import pickle
import runs
import seaborn as sns
from tqdm import tqdm
import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import hypers
import json
import sys
import plot_utils as plot
import matplotlib as mpl

mpl.rcParams["font.size"] = 24
mpl.rcParams["svg.fonttype"] = "none"

CONTINUING = "continuing"
EPISODIC = "episodic"
type_map = {
        "MinAtarBreakout": EPISODIC,
        "MinAtarFreeway": EPISODIC,
        "LunarLanderContinuous-v2": EPISODIC,
        "Bimodal3Env": CONTINUING,
        "Bimodal2DEnv": CONTINUING,
        "Bimodal1DEnv": CONTINUING,
        "BipedalWalker-v3": EPISODIC,
        "Hopper-v2": EPISODIC,
        "PuddleWorld-v1": EPISODIC,
        "MountainCar-v0": EPISODIC,
        "MountainCarShaped": EPISODIC,
        "MountainCarContinuous-v0": EPISODIC,
        "PendulumFixed-v0": CONTINUING,
        "Pendulum-v0": CONTINUING,
        "PendulumNoShaped-v0": CONTINUING,
        "PendulumNoShapedPenalty-v0": CONTINUING,
        "PendulumPenalty-v0": CONTINUING,
        "PositivePendulumPenalty-v0": CONTINUING,
        "Acrobot-v1": EPISODIC,
        "Walker2d": EPISODIC,
        "Swimmer-v2": EPISODIC,
        "CGW": EPISODIC,
        }

env = "CGW"

DATA_FILES = [
    f"./results/Sparse/SAC/{env}_SACresults/data.pkl",
    f"./results/Sparse/SACVAC/{env}_SACresults/data.pkl",
]


# DATA_FILES = [
#     f"./results/GreedyACNoEntReg/actor_rho_0_05/{env}_GreedyACNoEntRegresults/data.pkl",
#     f"./results/SACAutoEnt/{env}_SACresults/data.pkl",
#     f"results/EntReg/{env}_GreedyACresults/data.pkl",
# ]

# DATA_FILES = [
#     f"results/VACReparam/{env}_vacresults/data.pkl",
#     f"results/SACLikelihoodNoBaseline/{env}_SACresults/data.pkl",
#     f"results/SACAblations/{env}_SACresults/data.pkl",
# ]


fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()

DATA = []
for f in tqdm(DATA_FILES):
    with open(f, "rb") as infile:
        d = pickle.load(infile)

    # if f == f"results/SACAblations/{env}_SACresults/data.pkl":
    #     config1 = {
    #         "reparameterized": False,
    #         "soft_q": True,
    #         "double_q": True,
    #     }
    #     config2 = {
    #         "reparameterized": True,
    #         "soft_q": True,
    #         "double_q": True,
    #     }

    #     config1 = hypers.hold_constant(d, config1)[1]
    #     config2 = hypers.hold_constant(d, config2)[1]

    #     d1 = hypers.renumber(d, config1)
    #     d2 = hypers.renumber(d, config2)

    #     # DATA.append(d1)
    #     DATA.append(d2)
    #     continue

    # if f == f"results/EntReg/{env}_GreedyACresults/data.pkl":
    #     d = runs.after(d, 10)
    #     config = {
    #         "MountainCarContinuous-v0": {
    #             'actor_lr_scale': 1.0,
    #             'alpha': 10.0,
    #             'critic_lr': 0.001,
    #         },
    #         "Acrobot-v1": {
    #             'actor_lr_scale': 0.1,
    #             'alpha': 0.01,
    #             'critic_lr': 0.001,
    #         },
    #         "Pendulum-v0": {
    #             'actor_lr_scale': 0.1,
    #             'alpha': 10.0,
    #             'critic_lr': 0.01,
    #         },
    #     }[env]

    #     config = hypers.hold_constant(d, config)[1]
    #     d = hypers.renumber(d, config)

    DATA.append(d)

# Find best hypers
BEST_IND = []
for agent in DATA:
    best_hp = hypers.best(agent, to=-1)[0]
    BEST_IND.append(best_hp)

    ag_name = agent["experiment"]["agent"]["agent_name"]
    if "greedy" in ag_name.lower():
        # reparam = \
        #     agent["experiment"]["agent"]["parameters"]["reparameterized"][0]
        # if not reparam:
        print(ag_name, best_hp)
        pprint(agent["experiment_data"][best_hp]["agent_hyperparams"])
        print(hypers.get_performance(agent, best_hp).mean())
        for run in agent["experiment_data"][best_hp]["runs"]:
            print(run["random_seed"], run["train_episode_rewards"].mean())
        exit()

# Generate plot labels
labels = [
    "SAC",
    "VanillaAC",
]

# labels = [
#     "VAC (Reparam)",
#     "SAC (Likelihood)",
#     "SAC (Reparam)",
# ]

#     labels.append(legend)
CMAP = "tab10"
colours = list(sns.color_palette(CMAP, 8).as_hex())
colours = list(map(lambda x: [x], colours))
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=sns.color_palette(CMAP))

# Plot the mean + standard error
print("=== Plotting mean with standard error")
PLOT_TYPE = "train"
SOLVED = 0
TYPE = "online" if PLOT_TYPE == "train" else "offline"
best_ind = list(map(lambda x: [x], BEST_IND))

fig, ax = plot.episode_steps(
    DATA,
    PLOT_TYPE,
    best_ind,
    # [0]*len(best_ind),
    labels,
    # env_type=type_map[env].lower(),
    figsize=(16, 16),
    colours=colours,
    # fig=fig,
    # ax=ax,
)

ax.set_title(env)

ax.set_xlim(0, 100)
# ax.set_xlim(0, 95000)
# ax.set_xticks([0, 95000])
# ax.set_ylim([-1000, 1000])
# ax.set_xticklabels([0, 100000])
fig.savefig(f"{os.path.expanduser('~')}/{env}.png", bbox_inches="tight")
