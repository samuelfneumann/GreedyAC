# Import modules
import numpy as np
import environment
import experiment
import pickle
from utils import experiment_utils as exp_utils
import click
import json
from copy import deepcopy
import os
import utils.hypers as hypers
import socket


@click.command(help="Given an environment, agent name, and a range of " +
               "values, trains the agent on the environment for each " +
               "hyperparameter setting with index corresponding to each " +
               "value in the argument range of values. Values in the range " +
               "that are higher than the maximum hyperparameter settings " +
               "will wrap around and perform subsequent runs. For " +
               "example, if there are 10 hyperparameter settings and one " +
               "element in the argument range is 11, then this element " +
               "will correspond to the 2nd run of the first hyperparameter " +
               "setting")
@click.option("--env-json", help="Path to the environment json " +
              "configuration file",
              type=str, required=True)
@click.option("--agent-json", help="Path to the agent json configuration file",
              type=str, required=True)
@click.option("--index", type=int, required=False, help="The index " +
              "of the hyperparameter to run", default=1)
@click.option("--monitor", "-m", is_flag=True, help="Whether or not to " +
              "render the scene as the agent trains.", type=bool)
@click.option("--after", "-a", type=int, default=-1, help="How many " +
              "timesteps (training) should pass before " +
              "rendering the scene")
@click.option("--save-dir", type=str, default="./results", help="Which " +
              "directory to save the results file in", required=False)
def run(env_json, agent_json, index, monitor, after, save_dir):
    """
    Perform runs over hyperparameter settings.

    Performs the runs on the hyperparameter settings indices specified by
    range(start, stop step), with values over the total number of
    hyperparameters wrapping around to perform successive runs on the same
    hyperparameter settings. For example, if there are 10 hyperparameter
    settings and we run with hyperparameter settings 12, then this is the
    (12 // 10) = 1 run of hyperparameter settings 12 % 10 = 2, where runs
    are 0-based indexed.

    Parameters
    ----------
    env_json : str
        The path to the JSON environment configuration file
    agent_json : str
        The path to the JSON agent configuration file
    start : int
        The hyperparameter index to start the sweep at
    stop : int
        The hyperparameter index to stop the sweep at
    step : int
        The stepping value between hyperparameter settings indices
    monitor : bool
        Whether or not to render the scene as the agent trains
    after : int
        How many training + evaluation timesteps should pass before rendering
        the scene
    save_dir : str
        The directory to save the data in
    """
    # Read the config files
    with open(env_json) as in_json:
        env_config = json.load(in_json)
    with open(agent_json) as in_json:
        agent_config = json.load(in_json)

    main(agent_config, env_config, index, monitor, after, save_dir)


def main(agent_config, env_config, index, monitor, after,
         save_dir="./results"):
    """
    Runs experiments on the agent and environment corresponding the the input
    JSON files using the hyperparameter settings corresponding to the indices
    returned from range(start, stop, step).

    Saves a pickled python dictionary of all training and evaluation data.

    Note: this function will run the experiments sequentially.

    Parameters
    ----------
    agent_json : dict
        The agent JSON configuration file, as a Python dict
    env_json : dict
        The environment JSON configuration file, as a Python dict
    index : int
        The index of the hyperparameter setting to run
    monitor : bool
        Whether to render the scene as the agent trains or not
    after : int
        How many training + evaluation timesteps should pass before rendering
        the scene
    save_dir : str
        The directory to save all data in
    """
    # Create the data dictionary
    data = {}
    data["experiment"] = {}

    # Experiment meta-data
    data["experiment"]["environment"] = env_config
    data["experiment"]["agent"] = agent_config

    # Experiment runs per each hyperparameter
    data["experiment_data"] = {}

    # Calculate the number of timesteps before rendering. It is inputted as
    # number of training steps, but the environment uses training + eval steps
    if after >= 0:
        eval_steps = env_config["eval_episodes"] * \
            env_config["steps_per_episode"]
        eval_intervals = 1 + (after // env_config["eval_interval_timesteps"])
        after = after + eval_steps * eval_intervals
        print(f"Evaluation intervals before monitor: {eval_intervals}")

    # Get the directory to save in
    host = socket.gethostname()
    if not save_dir.startswith("./results"):
        save_dir = os.path.join("./results", save_dir)

    # If running on Compute Canada, then save in project directory
    if "computecanada" in host.lower():
        home = os.path.expanduser("~")
        save_dir = os.path.join(f"{home}/project/def-whitem/sfneuman/" +
                                "CEM-PyTorch", save_dir[2:])

    # Append name of environment and agent to the save directory
    save_dir = os.path.join(save_dir, env_config["env_name"] + "_" +
                            agent_config["agent_name"] + "results/")
    # Run the experiments
    # Get agent params from config file for the next experiment
    agent_run_params, total_sweeps = hypers.sweeps(
        agent_config["parameters"], index)
    agent_run_params["gamma"] = env_config["gamma"]

    print(f"Total number of hyperparam combinations: {total_sweeps}")

    # Calculate the run number and the random seed
    RUN_NUM = index // total_sweeps
    RANDOM_SEED = np.iinfo(np.int16).max - RUN_NUM

    # Create the environment
    env_config["seed"] = RANDOM_SEED
    if agent_config["agent_name"] == "linearAC" or \
       agent_config["agent_name"] == "linearAC_softmax":
        if "use_tile_coding" in env_config:
            use_tile_coding = env_config["use_tile_coding"]
            env_config["use_full_tile_coding"] = use_tile_coding
            del env_config["use_tile_coding"]

    env = environment.Environment(env_config, RANDOM_SEED, monitor, after)
    eval_env = environment.Environment(env_config, RANDOM_SEED)

    num_features = env.observation_space.shape[0]
    agent_run_params["feature_size"] = num_features

    # Set up the data dictionary to store the data from each run
    hp_sweep = index % total_sweeps
    if hp_sweep not in data["experiment_data"].keys():
        data["experiment_data"][hp_sweep] = {}
        data["experiment_data"][hp_sweep]["agent_hyperparams"] = \
            dict(agent_run_params)
        data["experiment_data"][hp_sweep]["runs"] = []

    SETTING_NUM = index % total_sweeps
    TOTAL_TIMESTEPS = env_config["total_timesteps"]
    MAX_EPISODES = env_config.get("max_episodes", -1)
    EVAL_INTERVAL = env_config["eval_interval_timesteps"]
    EVAL_EPISODES = env_config["eval_episodes"]

    # Store the seed in the agent run parameters so that batch algorithms
    # can sample randomly
    agent_run_params["seed"] = RANDOM_SEED

    # Include the environment observation and action spaces in the agent's
    # configuration so that neural networks can have the corrent number of
    # output nodes
    agent_run_params["observation_space"] = env.observation_space
    agent_run_params["action_space"] = env.action_space

    # Saving this data is redundant since we save the env_config file as
    # well. Also, each run has the run number as the random seed
    run_data = {}
    run_data["run_number"] = RUN_NUM
    run_data["random_seed"] = RANDOM_SEED
    run_data["total_timesteps"] = TOTAL_TIMESTEPS
    run_data["eval_interval_timesteps"] = EVAL_INTERVAL
    run_data["episodes_per_eval"] = EVAL_EPISODES

    # Print some data about the run
    print(f"SETTING_NUM: {SETTING_NUM}")
    print(f"RUN_NUM: {RUN_NUM}")
    print(f"RANDOM_SEED: {RANDOM_SEED}")
    print('Agent setting: ', agent_run_params)

    # Create the agent
    print(agent_config["agent_name"])
    agent_run_params["env"] = env
    agent = exp_utils.create_agent(agent_config["agent_name"],
                                   agent_run_params)

    # Initialize and run experiment
    exp = experiment.Experiment(
        agent,
        env,
        eval_env,
        EVAL_EPISODES,
        TOTAL_TIMESTEPS,
        EVAL_INTERVAL,
        MAX_EPISODES,
    )
    exp.run()

    # Save the agent's learned parameters, with these parameters and the
    # hyperparams, training can be exactly resumed from the end of the run
    run_data["learned_params"] = agent.get_parameters()

    # Save any information the agent saved during training
    run_data = {**run_data, **agent.info, **exp.info, **env.info}

    # Save data in parent dictionary
    data["experiment_data"][hp_sweep]["runs"].append(run_data)

    # After each run, save the data. Since data is accumulated, the
    # later runs will overwrite earlier runs with updated data.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_file = save_dir + env_config["env_name"] + "_" + \
        agent_config["agent_name"] + f"_data_{index}.pkl"

    print("=== Saving ===")
    print(save_file)
    print("==============")
    with open(save_file, "wb") as out_file:
        pickle.dump(data, out_file)


if __name__ == "__main__":
    run()
