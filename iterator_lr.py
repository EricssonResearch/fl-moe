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

            if (model == "cnn") and (config["dataset"] in ["cifar100"]):
                mylogger.warning("Skipping one CNN test")
                continue

            if (model == "leaf") and (config["dataset"] in ["cifar100"]):
                mylogger.warning("Skipping one LEAF test")
                continue

            pvals = np.logspace(-5,-3, 8)
            mylogger.info(f"Starting experiment with {model} from {filename} with lr={pvals}")

            child_processes = []

            # Make variable replacable
            for n, p in enumerate(pvals):

                config["model"] = model
                config["lr"] = p
                config["ft_lr"] = p
                config["local_lr"] = p
                config["moe_lr"] = p

                config["gpu"] = n % 8
                dataset = config["dataset"]
                config["filename"] = f"results_{dataset}_{model}_lr_{p}.csv"

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
