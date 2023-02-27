#!/usr/bin/env python3

import click
import utils.experiment_utils as exp
import utils.hypers as hypers
import json
import os
import pickle
import signal
import shutil
import sys
from tqdm import tqdm


signal.signal(signal.SIGINT, lambda: exit(0))


@click.command(help="Combine multiple data dictionaries into one")
@click.argument("save_file", type=click.Path())
@click.argument("data_files", nargs=-1)
def combine(save_file, data_files):
    if len(data_files) == 0:
        return

    # Create the save directory if it doesn't exist
    if os.path.dirname(save_file) != "" and not os.path.isdir(
       os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    if len(data_files) == 1:
        if os.path.isdir(data_files[0]):
            new_data_files = list(
                filter(
                    lambda x: not x.startswith("."),
                    os.listdir(data_files[0]),
                )
            )

            # Prepend the directory to the new data files
            dir_ = data_files[0]
            data_files = list(
                map(
                    lambda x: os.path.join(dir_, x),
                    new_data_files,
                )
            )
        else:
            # If only given a single data file, then we just copy it
            shutil.copy2(data_files[0], save_file)
            return

    # Read all input data dicts
    data = []
    for file in tqdm(data_files):
        print(file)
        with open(file, "rb") as infile:
            try:
                data.append(pickle.load(infile))
            except pickle.UnpicklingError:
                print("could not combine file:", file)

    # Get the configuration file
    # Remove the key `batch/replay` if present, since it is only used for
    # house-keeping when generating the hyper setting. The `batch` and `replay`
    # keys themselves will still exist in the config separately.
    config = data[0]["experiment"]["agent"]
    if "batch/replay" in config["parameters"]:
        del config["parameters"]["batch/replay"]

    # Combine all input data dicts
    new_data = hypers.combine(config, *data)

    # Write the new data file
    if os.path.isfile(save_file):
        print(f"file exists at {save_file}")
        print("Do you want to overwrite this file? (q/ctrl-c to cancel) ")
        overwrite = input()
        if overwrite.lower() == "q":
            exit(0)

    with open(save_file, "wb") as outfile:
        pickle.dump(new_data, outfile)


if __name__ == "__main__":
    combine()
