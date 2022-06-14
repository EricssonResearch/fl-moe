import subprocess
import json
from utils.util import read_config, get_logger
import numpy as np
import argparse
import torch.cuda as cutorch
from utils.gpuutils import get_available_gpus
import time

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', default=[], help='configuration filename',
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
        experiment = config.get("experiment","default")
        flags = config.pop("flags")
        # for clusters in range(1, config["clusters"] + 1  )

        epsvals = np.logspace(-2,np.log10(.5),10)

        frac = config["frac"]
        mylogger.info(f"Starting {experiment} from {filename} with eps={epsvals}")

        dataset = config["dataset"]
        model = config["model"]

        cluster_list = range(1, 5 + 1)

        # Make variable replacable
        for clusters in cluster_list:
            #config["frac"] = clusters * frac
            mylogger.info(f"Cluster k={clusters}")
            child_processes = []

            for n, eps in enumerate(epsvals):

                config["clusters"] = clusters
                config["eps"] = eps

                available_gpus = get_available_gpus()

                if not available_gpus:
                    time.sleep(60)

                config["gpu"] = np.random.choice(available_gpus, 1)[0]

                mylogger.debug(f"Assigning eps={eps} to GPU {gpus[n % number_of_gpus]}")

                config["filename"] = "results_clusters"

                command = ["python", "main_fed.py"]

                for k, v in config.items():
                    command.extend([f"--{k}", str(v)])

                for k, v in flags.items():
                    if v:
                        command.append(f"--{k}")

                mylogger.debug(" ".join(command))

                # Allow dry-runs
                if not args.dry_run:
                    p = subprocess.Popen(command, shell=False)
                    child_processes.append(p)

                time.sleep(20)

            for cp in child_processes:
                cp.wait()
