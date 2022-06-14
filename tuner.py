import main_fed
import os
import shutil
import argparse
from utils.util import read_config, get_logger
from utils.gpuutils import get_available_gpus
import numpy as np
import ray
from ray import tune

from random import randint
from time import sleep
from ray.tune.suggest.hyperopt import HyperOptSearch


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--filename', default=[], help='configuration filename',
        action="append")
    parser.add_argument('--dry-run', action='store_true', help='do not fire')
    parser.add_argument('--num_samples', type=int,
                        default=10, help="Number of samples")

    return parser.parse_args()


def get_fields(d):
    fields = {}
    for key, value in d.items():
        if isinstance(value, dict):
            fields.update(get_fields(value))
        else:
            fields[key] = value
    return fields


def train(config):

    # Avoid starting two experiments to close...

    logger = get_logger("tuner_train")
    sleeptime = randint(10, 60)
    logger.info(f"Sleeping for {sleeptime} seconds.")
    sleep(sleeptime)

    # Read config
    exp_config = read_config(config["config_filename"])
    exp_config.update(config)
    experiment = exp_config.pop("experiment")
    experiment_name = experiment.get("name", "default")
    flags = experiment["flags"]
    flags["iid"] = False
    exp_config.update(flags)
    exp_config["experiment"] = experiment_name + "_tune"
    exp_config["filename"] = experiment["filename"]
    exp_config["runs"] = experiment["runs"]
    exp_config["gpu"] = 0

    # Set up output paths
    log_path = f"save/{experiment_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # Copy experiment parameters for later reference
    shutil.copy2(exp_config["config_filename"],
                 os.path.join(log_path, "tuner.json"))

    d = get_fields(exp_config)

    logger.debug(d)

    args = argparse.Namespace(**d)
    logger.debug(args)

    val_acc_avg_locals, val_acc_avg_fedavg, val_acc_avg_e2e = main_fed.main(
        args)

    if config["part"] == "local":
        result = val_acc_avg_locals
    elif config["part"] == "fl":
        result = val_acc_avg_fedavg
    else:
        result = val_acc_avg_e2e

    logger.debug(result)
    tune.report(accuracy=result)


if __name__ == "__main__":

    args = args_parser()
    logger = get_logger("tuner_main")

    for filename in args.filename:
        best_trial = None
        tune_configs = [
            # {
            #     "local_lr": tune.loguniform(1e-4, 1e-3),
            #     "local_weight_decay": tune.loguniform(1e-4, 1e-1),
            #     "localdropout": tune.uniform(0.5, 0.8),
            #     "localhiddenunits1": tune.choice([128, 256, 512, 1024]),
            #     "localfilters1": tune.choice([16, 32, 64]),
            #     "localfilters2": tune.choice([32, 64, 128, 256]),
            #     "config_filename": os.path.join(os.getcwd(), filename),
            #     "epochs": 2,  # turn off FL
            #     "loc_epochs": 400,
            #     "moe_epochs": 2,
            #     "num_clients": 50,
            #     "eval_num_clients": 20,
            #     "part": "local"
            # },
            # {
            #     "lr": tune.loguniform(1e-3, 1e-1),
            #     "explore_strategy": "none",
            #     "fldropout": tune.uniform(0.2, 0.8),
            #     "flhiddenunits1": tune.choice([256, 512, 1024, 2048]),
            #     "flfilters1": tune.choice([32, 64, 128]),
            #     "flfilters2": tune.choice([32, 64]),
            #     "fl_weight_decay": tune.loguniform(1e-4, 1e-2),
            #     #"eps": tune.uniform(0, 0.4),
            #     #  "local_ep": tune.choice([3, 5]),
            #     #  "local_bs": tune.choice([5, 10]),
            #     "config_filename": os.path.join(os.getcwd(), filename),
            #     "epochs": 500,
            #     "loc_epochs": 2,
            #     "moe_epochs": 2,
            #     # "num_clients": 24,
            #     # "frac": 0.25,
            #     "part": "fl",
            #     "clusters": 1
            # },
            {
                "moe_lr": tune.loguniform(1e-7, 1e-3),
                #"explore_strategy": "none",
                "gatedropout": tune.uniform(0.2, 0.8),
                "gatehiddenunits1": tune.choice([4, 8, 16, 32, 64]),
                #"clusters": 1,
                #"gatehiddenunits2": tune.choice([64, 128, 256, 512]),
                "gatefilters1": tune.choice([2, 4, 6, 8, 12, 16]),
                "gatefilters2": tune.choice([0, 2, 4, 6, 8, 12, 16]),
                "gate_weight_decay": tune.loguniform(1e-4, 1e-1),
                "config_filename": os.path.join(os.getcwd(), filename),
                "epochs": 100,
                "loc_epochs": 200,
                "moe_epochs": 200,
                # "num_clients": 50,
                # "frac": 0.25,
                "part": "moe"
            }
            # ,
            # {
            #     "explore_strategy": "eps",
            #     #"config_filename": os.path.join(os.getcwd(), filename),
            #     "eps": tune.uniform(0, 0.3),
            #     "p": 0.9,
            #     "clusters": 2,
            #     "part": "moe"
            # }
        ]

        best_trial_configs = []

        for tc in tune_configs:

            # ray.init()

            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, get_available_gpus()))
            logger.debug(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

            # Update the config with the best settings from the last search.
            if best_trial:
                btc = best_trial.config
                entries_to_remove = ["epochs", "loc_epochs", "moe_epochs",
                                     "part", "num_clients", "frac"]
                for k in entries_to_remove:
                    btc.pop(k, None)
                tc.update(btc)

            # TODO: This is just for MoE
            current_best_params = [{
                "moe_lr": 0.00002,
                "gatedropout": 0.7,
                "gatehiddenunits1": 4,
                "gatefilters1": 4,
                "gatefilters2": 2,
                "gate_weight_decay": 0.0003
            }]

            hyperopt = HyperOptSearch(
                metric="accuracy", mode="max",
                points_to_evaluate=current_best_params)

            result = tune.run(
                train,
                resources_per_trial={"cpu": 2, "gpu": 0.3},
                search_alg=hyperopt,
                config=tc,
                num_samples=args.num_samples,
                max_failures=args.num_samples // 2)

            best_trial = result.get_best_trial("accuracy", "max", "last")
            logger.info("Best trial config: {}".format(best_trial.config))
            logger.debug("Best trial final validation accuracy: {}".format(
                best_trial.last_result["accuracy"]))

            best_trial_configs.append(best_trial.config)

            # ray.stop()

        logger.info(best_trial_configs)
