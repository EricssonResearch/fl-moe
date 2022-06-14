import json
from utils.util import read_config, get_logger
import numpy as np
import argparse
#from utils.gpuutils import get_available_gpus
import time
import os
import shutil
import pprint
from k8s_helper import *
from kubernetes import client, config


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', default=[], help='configuration filename',
        action="append")
    parser.add_argument('--dry-run', action='store_true', help='do not fire')
    parser.add_argument('--min_clusters', type=int, default=1,
                        help="Minimum number of clusters to try")
    parser.add_argument('-j', nargs='+', type=int)
    parser.add_argument('--max_clusters', type=int, default=5,
                        help="Minimum number of clusters to try")
    parser.add_argument('--runs', type=int, default=1,
                        help="Minimum number of clusters to try")
    parser.add_argument('--splits', type=int, default=1,
                        help="Split job into parts")
    parser.add_argument('--name', type=str, required=True)
    return parser.parse_args()


def get_fields(d):
    fields = []
    for key, value in d.items():
        if isinstance(value, dict):
            fields.extend(get_fields(value))
        else:
            fields.extend([f"--{key}", str(value)])
    return fields


if __name__ == "__main__":

    config.load_kube_config()
    batch_v1 = client.BatchV1Api()

    args = args_parser()

    mylogger = get_logger("Iterator")

    mylogger.debug(args)

    user = "eisamar"
    gen_name = f"{user}-{args.name}"

    for filename in args.filename:

        mylogger.info(f"Starting experiment from {filename}")

        basename = os.path.splitext(os.path.basename(filename))[
            0].replace("config_", "").replace("_", "-")

        raw_clusters = np.arange(args.min_clusters, args.max_clusters + 1)
        if args.j is not None:
            raw_clusters = np.array(args.j)

        # eps_decay_b
        for run in range(args.runs):
            for strategy in ["none", "eps"]:
                for clusters in np.array_split(raw_clusters, args.splits):

                    if len(clusters) == 1:
                        c = str(clusters[0])
                    else:
                        c = " ".join(map(str, clusters))

                    command = ["python",
                               "iterator_clusters_old.py",
                               "--min_clusters", str(min(clusters)),
                               "--max_clusters", str(max(clusters)),
                               "--explore_strategy", strategy,
                               "--filename", filename]

                    job_name_r = f"{gen_name}-{basename}-{strategy}-{run}-"
                    mylogger.debug(job_name_r + " - " + " ".join(command))

                    # Allow dry-runs
                    if not args.dry_run:

                        create_job(
                            batch_v1,
                            create_job_object(command, gen_name=job_name_r))

                    #gpu_count += 1
