import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np
import argparse
import torch.cuda as cutorch

def get_available_gpus(threshold=0.5):
    """
    TODO: Implement
    """
    avail_gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    return avail_gpus


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename',
        default=[], help='configuration filename',
        action="append")
    parser.add_argument('--dry-run', action='store_true', help='do not fire')
    return parser.parse_args()


if __name__ == "__main__":



    args = args_parser()
    mylogger = get_logger("Iterator")

    mylogger.debug(args)
    # Loop over multiple files

    gpus = get_available_gpus()
    number_of_gpus = len(gpus)
    mylogger.debug(f"gpus: {gpus}")

    for filename in args.filename:
        config = read_config(filename)

        # for clusters in range(1, config["clusters"] + 1  )
        pvals = np.linspace(.2, 1, 9)
        mylogger.info(f"Starting experiment from {filename} with p={pvals}")

        child_processes = []

        dataset = config["dataset"]
        model = config["model"]

        # Make variable replacable
        for n, p in enumerate(pvals):

            config["p"] = np.round(p / .1) * .1
            config["gpu"] = gpus[n % number_of_gpus]
            mylogger.debug(f"Assigning p={p} to GPU {gpus[n % number_of_gpus]}")

            config["filename"] = f"results_{dataset}_{model}_p_{p:.2f}"

            command = ["python", "main_fed.py"]

            for k, v in config.items():
                command.extend([f"--{k}", str(v)])

            # command.extend(["--overlap"])
            mylogger.debug(command)

            # Allow dry-runs
            if not args.dry_run:
                p = subprocess.Popen(command)
                child_processes.append(p)

        for cp in child_processes:
            cp.wait()
