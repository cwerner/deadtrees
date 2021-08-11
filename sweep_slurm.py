import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import File

import yaml

import wandb

# Set API key
if Path("keys.json").is_file():
    with open("keys.json") as file:
        api_key = json.load(file)["wandb_key"]
        os.environ["WANDB_API_KEY"] = api_key

# Gather nodes allocated to current slurm job
result = subprocess.run(["scontrol", "show", "hostnames"], stdout=subprocess.PIPE)
node_list = result.stdout.decode("utf-8").split("\n")[:-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_config", type=argparse.FileType)
    parser.add_argument("train_script", type=argparse.FileType)
    parser.add_argument("project", type=str)

    args = parser.parse_args()

    wandb.init(project=args.project)

    with open(args.sweep_config) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    config_dict["program"] = args.train_script

    sweep_id = wandb.sweep(config_dict, project=args.project)

    sp = []
    for node in node_list:
        sp.append(
            subprocess.Popen(
                [
                    "srun",
                    "--nodes=1",
                    "--ntasks=1",
                    "-w",
                    node,
                    "start-agent.sh",
                    sweep_id,
                    args.project,
                ]
            )
        )
    exit_codes = [p.wait() for p in sp]  # wait for processes to finish
    return exit_codes


if __name__ == "__main__":
    main()
