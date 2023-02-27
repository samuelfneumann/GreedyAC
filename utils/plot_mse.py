# Script to plot mean learning curves with standard error
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

# ########################################################################
# To produce learning curves, simply fill in the following:
# Specify here the name of the environment you are plotting
env = "Pendulum-v0"


# Specify whether to plot online/training data or offline/evaluation data
PLOT_TYPE = "train"

# Specify here the data files to plot
DATA_FILES = [
    "./results/Pendulum-v0_GreedyACresults/data.pkl"
]

# Specify the labels for each data file. The length of labels and DATA_FILES
# should be equal
labels = [
    "GreedyAC",
]

# Lower and upper bounds on the x and y-axes of the resulting plot. Set to None
# to use default axis bounds
x_low, x_high = None, None
y_low, y_high = None, None

# Colours to plot with
colours = ["black", "red", "blue", "gold"]

# Directory to save the plot at
save_dir = os.path.expanduser('~')

# Types of files to save
filetypes = ["png", "svg", "pdf"]
# ########################################################################

mpl.rcParams["font.size"] = 24
mpl.rcParams["svg.fonttype"] = "none"

# Configuration stuff
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

colours = list(map(lambda x: [x], colours))

# Plot the mean + standard error
print("=== Plotting mean with standard error")
PLOT_TYPE = "train"
TYPE = "online" if PLOT_TYPE == "train" else "offline"
best_ind = list(map(lambda x: [x], BEST_IND))

fig, ax = plot.mean_with_err(
    DATA,
    PLOT_TYPE,
    best_ind,
    [0]*len(best_ind),
    labels,
    env_type=type_map[env].lower(),
    figsize=(12, 12),
    colours=colours,
    skip=-1,  # The number of data points to skip when plotting
)

ax.set_title(env)
if x_low is not None and x_high is not None:
    ax.set_xlim(x_low, x_high)
if y_low is not None and y_high is not None:
    ax.set_ylim(y_low, y_high)

for filetype in filetypes:
    print(f"Saving at {save_dir}/{env}.{filetype}")
    fig.savefig(f"{save_dir}/{env}.{filetype}", bbox_inches="tight")
