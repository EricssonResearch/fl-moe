import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np
import argparse


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

    for filename in args.filename:
        config = read_config(filename)
        models = ["cnn", "leaf"]

        for model in models:

            optvals = np.linspace(0, .9, 10)
            pvals = [0.2, 0.5, 0.8]
            mylogger.info(f"Starting experiment with {model} from {filename} with opt={optvals}")

            child_processes = []

            # Make variable replacable
            for m, pv in enumerate(pvals):
                for n, opt in enumerate(optvals):

                    config["model"] = model
                    config["opt"] = opt
                    config["p"] = pv
                    config["gpu"] = n % 8
                    dataset = config["dataset"]
                    config["filename"] = f"results_{dataset}_{model}_opt_{opt}_p_{pv}.csv"

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
