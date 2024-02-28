# Greedy AC

The official codebase for the paper _Greedy Actor-Critic: A New Conditional
Cross-Entropy Method for Policy Improvement_.

## Installing Dependencies

To install dependencies, run:

```
pip install -r requirements.txt
```

## Running the Code

This codebase works by reading `json` configuration files and running the
experiments which are outlined by those configuration files. When running an
experiment, you need to specify two configuration files, one for the algorithm
and one for the environment. Examples configuration files for algorithms and
environments can be found in `./config/agent` and `./config/environment`
respectively.

Experiments are run using the `main.py` file. To run this file, simply use the
following code examples:

```bash
python3 main.py --agent-json AGENT_JSON --env-json ENV_JSON --index INDEX --save-dir SAVE_DIR
```

where `AGENT_JSON` is the path to an algorithm's configuration file, `ENV_JSON`
is the path to an environment's configuration file, and `INDEX` is an integer
representing the hyperparameter setting in the agent configuration file to use.
Data from the experiment will be saved at `./results/SAVE_DIR`.

For more information, see `python3 main.py --help`

### Combining Mutiple Outputs

When you run many experiments, you may end up with many different data files.
For example, if you run

```bash
for i in $(seq 0 2); do
	python3 main.py --agent-json config/agent/GreedyAC.json --env-json config/enviroment/AcrobotContinuous-v1.json --index 0 --save-dir "output"
done
```

You'll end up with three files in the `output` directory:

```
AcrobotContinuous-v1_GreedyAC_data_0.pkl
AcrobotContinuous-v1_GreedyAC_data_1.pkl
AcrobotContinuous-v1_GreedyAC_data_2.pkl
```

To combine all these files into one, you can do the following:

```bash
./combine.py output/combined.pkl output/
```

This will combine all the files in output to produce a single `combined.pkl`
file. If you run `ls output`, you'll see:

```
AcrobotContinuous-v1_GreedyAC_data_0.pkl
AcrobotContinuous-v1_GreedyAC_data_1.pkl
AcrobotContinuous-v1_GreedyAC_data_2.pkl
combined.pkl
```

You can now safely delete the three individual data files, as they have been
combined into the single `combined.pkl` data file.

## Configuration Files

### Environment Configuration Files

Environment configuration files describe the environment to use for an
experiment. For example, here is an environment configuration file for the
`MountainCarContinuous-v0` environment:

```json
{
    "env_name": "MountainCarContinuous-v0",
    "total_timesteps": 100000,
    "steps_per_episode": 1000,
    "eval_interval_timesteps": 10000,
    "eval_episodes": 5,
    "gamma": 0.99,
    "overwrite_rewards": false,
    "continuous": true,
    "rewards": {},
    "start_state": []
}
```

This configuration file specifies that we should run an experiment for 100,000
timesteps, with episodes cut off at 1,000 timesteps. We will run offline
evaluation every 10,000 steps using 5 episodes. Online evaluation is always
recorded, but offline evaluation may not be recorded if `eval_episodes = 0` or
`eval_interval_timesteps > total_timesteps`.

It is also possible to override the stating state of the environment and
override the environment's rewards. For example, to create a cost-to-goal
version of MountainCarContinuous-v0, we could set `overwrite_rewards = true`
and `rewards = {"goal": -1, "timestep": -1}`.

### Algorithm Configuration File

Algorithm configuration files are a bit more complicated than environment
configuration files. These configuration files can be found in
`./config/agent`.

An example algorithm configuration file is:

```json
{
    "algo_name": "example_algo",
    "hyper1": [1, 2, 3],
	"hyper2": [4, 5, 6]
}
```

This configuration file outlines 9 different hyperparameter configurations for
algorithm `example_algo`, one hyperparameter setting for each combination of
"hyper1" and "hyper2". We refer to these combinations with 0-based indexing:

- index 0 has `hyper1 = 1` and `hyper2 = 4`
- index 1 has `hyper1 = 2` and `hyper2 = 4`
- index 2 has `hyper1 = 3` and `hyper2 = 4`
- index 3 has `hyper1 = 1` and `hyper2 = 5`
- index 4 has `hyper1 = 2` and `hyper2 = 5`
- index 5 has `hyper1 = 3` and `hyper2 = 5`
- index 6 has `hyper1 = 1` and `hyper2 = 6`
- index 7 has `hyper1 = 2` and `hyper2 = 6`
- index 8 has `hyper1 = 3` and `hyper2 = 6`

When running experiments from the command line, you specify this index using
the `--index` option. A single run of the experiment will then be executed
using the associated hyperparameter setting. To run multiple hyperparameter
settings in parallel, you can use [GNU
Parallel](https://www.gnu.org/software/parallel/)

One data file will be saved each time you run an experiment. To combine these
individual data files into a single data file, you can use the `combine.py`
script:

```bash
python3 combine.py SAVE_FILE PATH_TO_DATA_DIR
```

where `PATH_TO_DATA_DIR` is the path to the directory holding all the
individual data files to combine and `SAVE_FILE` is the desired path/filename
of the resulting file combined data file.

## Citing

If you use this code or reference our work, please cite:
```
@inproceedings{neumann2023greedy,
	title = {Greedy Actor-Critic: A New Conditional Cross-Entropy Method for
	Policy Improvement},
	author = {Neumann, Samuel and Lim, Sungsu and Joseph, Ajin and Yangchen, Pan and White, Adam and White, Martha},
	year = {2023}
	journal = {International Conference on Learning Representations},
}
```
