import argparse
import subprocess
import random
import numpy as np
from Pinnoloc.utils.experiments import read_yaml_to_dict
from Pinnoloc.train import train


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run experiments based on YAML configuration.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Read hyperparameters from the YAML file
    config = read_yaml_to_dict(args.config)

    # get the setting file path from the config file path
    setting_path = '/'.join(args.config.split('/')[:-2]) + '/setting.yaml'
    setting = read_yaml_to_dict(setting_path)

    # try to run train(setting, config)
    try:
        train(setting, config)
    except Exception as e:
        print(f"Training failed with error: {e}.")


if __name__ == "__main__":
    main()
