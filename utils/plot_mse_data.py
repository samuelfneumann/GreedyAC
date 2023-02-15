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
        "MountainCar-v0": EPISODIC,
        "MountainCarContinuous-v0": EPISODIC,
        "Pendulum-v0": CONTINUING,
        "Acrobot-v1": EPISODIC,
        "Swimmer-v2": EPISODIC,
        }

env = "Pendulum-v0"

DATA_FILES = [
    f"./results/SAC/{env}_SACresults/data.pkl",
    f"./results/GreedyAC/{env}_SACresults/data.pkl",
]

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot()

DATA = []
for f in tqdm(DATA_FILES):
    with open(f, "rb") as infile:
        d = pickle.load(infile)
    DATA.append(d)

# Find best hypers
BEST_IND = []
for agent in DATA:
    best_hp = hypers.best(agent, to=-1)[0]
    BEST_IND.append(best_hp)

# Generate plot labels
labels = [
    "SAC",
    "GreedyAC",
]

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
    labels,
    figsize=(16, 16),
    colours=colours,
)

ax.set_title(env)
ax.set_xlim(0, 100)
fig.savefig(f"{os.path.expanduser('~')}/{env}.png", bbox_inches="tight")
