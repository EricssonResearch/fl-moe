"""
Federated learning using a mixture of experts

based on https://github.com/edvinli/federated-learning-mixture
"""

import os.path
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import sys
import tempfile


from utils.sample_data import mnist_iid, mnist_iid2, mnist_noniid2, \
    cifar_iid, cifar_iid2, cifar_noniid, cifar_noniid2

from utils.arguments import args_parser
from models.ClientUpdate import ClientUpdate

from models.Models import MLP, CNNCifar, GateCNN, GateMLP, CNNFashion, \
    GateCNNFashion, CNNLeaf, \
    CNNLeafFEMNIST, GateCNNFEMNIST, GateCNNLeaf, CNNIFCA
from models.Models import MyEnsemble

from models.FederatedAveraging import FedAvg
from models.test_model import test_img, test_img_mix

from utils.util import get_logger
import json
import uuid

from torch.utils.tensorboard import SummaryWriter


def rename_keys(d):
    """
    For a one-level dictionary, return a new dictionary with keys as numbers
    """
    return {n: v for n, (k, v) in enumerate(d.items())}


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.xavier_uniform(m.bias.data)
    elif isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def do_explore(iteration, args):

    if args.explore_strategy == "eps":
        return np.random.random() < args.eps

    if args.explore_strategy == "eps_decay":
        return np.random.random() < 1.0 / (iteration + 1)

    if args.explore_strategy == "eps_decay_b":
        b = -np.log(args.eps) / np.log((args.epochs / 8) + 1)
        return np.random.random() < 1.0 / ((iteration + 1)**b)

    if args.explore_strategy == "eps_decay_k":
        return np.random.random() < 1.0 / ((iteration + 1)**(2 / args.clusters))

    return False


def main(args):

    # A short UUID, perhaps not collision free but close enough
    myid = str(uuid.uuid4())[:8]
    mylogger = get_logger(myid)

    # Set up logging to file
    if not os.path.exists(f"save/{args.experiment}"):
        os.makedirs(f"save/{args.experiment}", exist_ok=True)

    filename = args.filename
    filexist = os.path.isfile(f'save/{args.experiment}/{filename}.csv')

    fields = ["client_id", "dataset", "model", "epochs", "local_ep",
              "num_clients", "iid",
              "p", "opt", "n_data", "train_frac", "train_gate_only",
              "val_acc_avg_e2e", "val_acc_avg_e2e_neighbour",
              "val_acc_avg_locals",
              "val_acc_avg_fedavg", "ft_val_acc", "val_acc_avg_3",
              "val_acc_avg_rep", "val_acc_avg_repft", "val_acc_avg_ensemble",
              "acc_test_mix", "acc_test_locals", "acc_test_fedavg",
              "ft_test_acc", "ft_train_acc", "train_acc_avg_locals",
              "val_acc_gateonly", "overlap", "run", "clusters", "eps",
              "explore_strategy", "best_iteration"]

    if not filexist:
        with open(f"save/{args.experiment}/{myid}_{filename}.csv", 'a') as f1:
            f1.write(";".join(fields))
            f1.write('\n')

    trans_cifar10_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            mean=(0.4915, 0.4822, 0.4466),
            std=(0.2470, 0.2435, 0.2616))
         ])

    trans_cifar100_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(
            mean=(0.4915, 0.4822, 0.4466),
            std=(0.2470, 0.2435, 0.2616))
         ])

    if args.dataaugmentation:
        trans_cifar10_train = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.ColorJitter(
                    brightness=.4, contrast=.4, saturation=.4, hue=.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4915, 0.4822, 0.4466),
                    std=(0.2470, 0.2435, 0.2616))
            ])

        trans_cifar100_train = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.ColorJitter(
                    brightness=.4, contrast=.4, saturation=.4, hue=.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4915, 0.4822, 0.4466),
                    std=(0.2470, 0.2435, 0.2616))
            ])

    else:
        trans_cifar10_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4915, 0.4822, 0.4466),
                    std=(0.2470, 0.2435, 0.2616))
            ])

        trans_cifar100_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4915, 0.4822, 0.4466),
                    std=(0.2470, 0.2435, 0.2616))
            ])

    # TODO: print warnings if arguments are not used (p, overlap)
    for run in range(args.runs):

        args.device = torch.device('cuda:{}'.format(args.gpu))

        # Create datasets TODO: Refactor
        # TODO: Remove to device?
        if args.dataset == 'mnist':

            trans_mnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True,
                download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False,
                download=True, transform=trans_mnist)

            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_clients)
            else:
                dict_users = mnist_noniid2(
                    dataset_train, args.num_clients, args.p)

        elif args.dataset == "femnist":

            from FemnistDataset import FemnistDataset

            # TODO: add transform
            # trans_femnist = transforms.Compose(
            #    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

            # TODO: remove absolute path
            root_dir = "/proj/second-carrier-prediction/leaf/data/femnist/data/"
            dataset_train = FemnistDataset(root_dir=root_dir, train=True)
            dataset_test = FemnistDataset(root_dir=root_dir, train=False)

            # TODO: Add user sampling for the population
            dict_users = rename_keys(dataset_train.dict_users)
            dict_users_test = rename_keys(dataset_test.dict_users)

        elif args.dataset == 'cifar10':

            with tempfile.TemporaryDirectory() as tmpdirname:
                dataset_train = datasets.CIFAR10(
                    tmpdirname, train=True,
                    download=True,
                    transform=trans_cifar10_train)
                dataset_test = datasets.CIFAR10(
                    tmpdirname, train=False,
                    download=True,
                    transform=trans_cifar10_test)

            if args.iid:
                dict_users = cifar_iid(
                    dataset_train, args.num_clients, args.n_data)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients,
                    args.p, args.n_data, args.n_data_test, args.overlap)

        elif args.dataset == "cifar10rot":

            from Cifar10RotatedDataset import Cifar10RotatedDataset

            with tempfile.TemporaryDirectory() as tmpdirname:

                dataset_train = Cifar10RotatedDataset(
                    tmpdirname, train=True,
                    download=True, transform=trans_cifar10_train,
                    num_clients=args.num_clients,
                    n_data=args.n_data)

                dataset_test = Cifar10RotatedDataset(
                    tmpdirname, train=False,
                    download=True, transform=trans_cifar10_test,
                    num_clients=args.num_clients,
                    n_data=args.n_data_test)

            dict_users = dataset_train.dict_users
            dict_users_test = dataset_test.dict_users

        elif args.dataset == 'cifar100':

            dataset_train = datasets.CIFAR100(
                '../data/cifar100', train=True, download=True,
                transform=trans_cifar100_train)
            dataset_test = datasets.CIFAR100(
                '../data/cifar100', train=False, download=True,
                transform=trans_cifar100_test)

            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients, args.p,
                    args.n_data, args.n_data_test, args.overlap)

        elif args.dataset == 'fashion-mnist':

            trans_fashionmnist = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST(
                '../data/fashion-mnist', train=True, download=True,
                transform=trans_fashionmnist)
            dataset_test = datasets.FashionMNIST(
                '../data/fashion-mnist', train=False, download=True,
                transform=trans_fashionmnist)

            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_clients)
            else:
                dict_users, dict_users_test = cifar_noniid2(
                    dataset_train, dataset_test, args.num_clients, args.p,
                    args.n_data, args.n_data_test, args.overlap)

        else:
            mylogger.error("Dataset not available")
            raise SystemExit(3)

        train_lengths = [len(v) for k, v in dict_users.items()]
        mylogger.debug(f"Training samples: {train_lengths}")
        test_lengths = [len(v) for k, v in dict_users_test.items()]
        mylogger.debug(f"Test samples: {test_lengths}")

        img_size = dataset_train[0][0].shape
        mylogger.debug(f"Sample size: {img_size}")

        input_length = 1
        for x in img_size:
            input_length *= x

        gates_e2e = []
        net_locals = []

        # TODO: Remove sending to device here?
        if args.model == 'cnn':

            if args.dataset in ['cifar10', 'cifar100', "cifar10rot"]:

                net_glob_fedAvg = CNNCifar(args=args).to(args.device)
                gates_e2e_model = GateCNN(args=args).to(args.device)
                net_locals_model = CNNCifar(args=args).to(args.device)

            elif args.dataset in ['mnist', 'fashion-mnist', "femnist"]:

                net_glob_fedAvg = CNNFashion(args=args).to(args.device)
                gates_e2e_model = GateCNNFashion(args=args).to(args.device)
                net_locals_model = CNNFashion(args=args).to(args.device)

        elif args.model == 'leaf':

            if "cifar10" in args.dataset:

                net_glob_fedAvg = CNNLeaf(
                    args=args, model="fl").to(args.device)
                gates_e2e_model = GateCNNLeaf(args=args).to(args.device)
                net_locals_model = CNNLeaf(
                    args=args, model="local").to(args.device)

            elif "mnist" in args.dataset:

                net_glob_fedAvg = CNNLeafFEMNIST(args=args).to(args.device)
                gates_e2e_model = GateCNNFEMNIST(args=args).to(args.device)
                net_locals_model = CNNLeafFEMNIST(args=args).to(args.device)

        elif args.model == 'ifca':

            if "cifar10" in args.dataset:

                net_glob_fedAvg = CNNIFCA(
                    args=args, model="fl").to(args.device)
                gates_e2e_model = GateCNNLeaf(args=args).to(args.device)
                net_locals_model = CNNIFCA(
                    args=args, model="local").to(args.device)

            else:

                mylogger.error(f"No model implemented for {args.dataset}.")
                raise SystemExit(2)

        elif args.model == 'mlp':

            net_glob_fedAvg = MLP(dim_in=input_length,
                                  dim_hidden=200,
                                  dim_out=args.num_classes).to(args.device)
            gates_e2e_model = GateMLP(dim_in=input_length,
                                      dim_hidden=200,
                                      dim_out=1).to(args.device)
            net_locals_model = MLP(dim_in=input_length,
                                   dim_hidden=200,
                                   dim_out=args.num_classes).to(args.device)

        else:
            mylogger.error("No such model.")
            raise SystemExit(2)

        # Initialize weights
        net_glob_fedAvg.apply(weights_init)
        gates_e2e_model.apply(weights_init)
        net_locals_model.apply(weights_init)

        # opt-out fraction
        opt = np.ones(args.num_clients)
        opt_out = np.random.choice(range(args.num_clients), size=int(
            args.opt * args.num_clients), replace=False)
        opt[opt_out] = 0.0

        # TODO: Same starting weights for local models, good?
        for i in range(args.num_clients):
            gates_e2e.append(copy.deepcopy(gates_e2e_model))
            net_locals.append(copy.deepcopy(net_locals_model))

        # Initialize cluster models
        mylogger.info(f"Initializing {args.clusters} cluster models.")

        net_clusters = []
        for k in range(args.clusters):
            net_clusters.append(copy.deepcopy(net_glob_fedAvg))
            net_clusters[-1].apply(weights_init)

        # TODO: Remove? Since train is called in ClientUpdate
        for i in range(args.num_clients):
            gates_e2e[i].train()
            net_locals[i].train()

        # training
        acc_test_locals, acc_test_mix, acc_test_fedavg = [], [], []
        acc_test_finetuned_avg = []

        mylogger.info(f"Starting Federated Learning with {args.num_clients} clients for {args.epochs} rounds.")

        if args.tensorboard:
            tb_writers = [SummaryWriter(f"save/{args.experiment}/{myid}/fl/{c}") for c in range(args.clusters)]
        else:
            tb_writers = [None for c in range(args.clusters)]

        patience = 10
        cluster_counter = [0]*args.clusters
        cluster_val_loss_best = [np.inf]*args.clusters
        w_fedAvg_best = {}
        cluster_model_max_iteration = [0] * args.clusters
        best_iteration = args.epochs

        for iteration in range(args.epochs):
            mylogger.info(f"Round {iteration}")

            w_fedAvg = {c: [] for c in range(args.clusters)}
            cluster_train_loss = {c: [] for c in range(args.clusters)}
            cluster_val_loss = {c: [] for c in range(args.clusters)}
            cluster_val_acc = {c: [] for c in range(args.clusters)}
            cluster_clients = {c: 0 for c in range(args.clusters)}
            alpha = {c: [] for c in range(args.clusters)}

            m = max(np.ceil(args.frac * args.num_clients), 1).astype(int)

            idxs_users = np.random.choice(
                range(args.num_clients), m, replace=False)

            for idx in idxs_users:
                mylogger.debug(f"FedAvg client {idx}")

                client = ClientUpdate(args=args,
                                      train_set=dataset_train,
                                      val_set=dataset_test,
                                      idxs_train=dict_users[idx],
                                      idxs_val=dict_users_test[idx],
                                      parent_id=myid,
                                      client_id=idx)

                # TODO: Affects C. This means that in some cases, _only_
                # opt-out clients can be selected.

                if(opt[idx]):
                    # train FedAvg

                    # 1. Find best fitting cluster based on training set
                    # TODO: Log cluster assignments
                    # TODO: Refactor, make another way of choosing cluster based on MoE?
                    # TODO: Add epsilon, if r < eps, randomly assign.

                    c_loss = []
                    for c in range(args.clusters):

                        # Get the loss from the training set.
                        _, cluster_loss_fed = client.validate(
                            net=copy.deepcopy(
                                net_clusters[c]).to(args. device),
                            train=True)
                        c_loss.append(cluster_loss_fed)

                        # cluster_train_loss[c].append(cluster_loss_fed)

                    if args.clusters > 1:

                        # Sometimes randomly pick one
                        if do_explore(iteration, args):

                            c_idx = np.random.randint(args.clusters)

                        else:
                            # Returns all indicies
                            c_indicies = np.where(
                                c_loss == np.nanmin(c_loss))

                            # If more than one, pick one on random
                            try:
                                c_idx = np.random.choice(c_indicies[0], 1)[0]
                            except ValueError:
                                c_idx = np.random.randint(args.clusters)

                        mylogger.debug(f"Client {idx} chose cluster {c_idx}.")

                    else:
                        c_idx = 0

                    # 2. Start with that cluster model
                    w_glob_fedAvg, train_loss_fed = client.train(net=copy.deepcopy(
                        net_clusters[c_idx]).to(args.device), n_epochs=args.local_ep,
                        offset=iteration * args.local_ep,
                        weight_decay=args.fl_weight_decay)

                    cluster_train_loss[c_idx].append(train_loss_fed)

                    # 3. Update that cluster
                    cluster_clients[c_idx] += 1

                    w_fedAvg[c_idx].append(copy.deepcopy(w_glob_fedAvg))

                    # Weigh models by client dataset size
                    # 4.
                    alpha[c_idx].append(
                        len(dict_users[idx]) / len(dataset_train))

                # Don't evaluate every iteration
                if iteration % 10 == 0:

                    # Get the loss from the validation set.
                    val_acc_fed, val_loss_fed = client.validate(
                        net=copy.deepcopy(
                            net_clusters[c_idx]).to(args. device),
                        train=False)

                    cluster_val_loss[c_idx].append(val_loss_fed)
                    cluster_val_acc[c_idx].append(val_acc_fed)

            # update global model weights
            for c in range(args.clusters):

                if iteration % 10 == 0:

                    if tb_writers[c]:
                        tb_writers[c].add_scalar('fl training loss', np.mean(
                            cluster_train_loss[c]), iteration)
                        tb_writers[c].add_scalar('fl validation loss', np.mean(
                            cluster_val_loss[c]), iteration)
                        tb_writers[c].add_scalar(
                            'fl number of clients', cluster_clients[c], iteration)

                if not w_fedAvg[c]:
                    mylogger.warning(f"Empty FL gradient list for cluster {c} in round {iteration}.")
                else:
                    w_glob_fedAvg = FedAvg(w_fedAvg[c], alpha[c])

                    # copy weight to net_glob
                    net_clusters[c].load_state_dict(w_glob_fedAvg)

                if iteration % 10 == 0:

                    if np.mean(cluster_val_loss[c]) < cluster_val_loss_best[c]:
                        cluster_counter[c] = 0
                        cluster_val_loss_best[c] = np.mean(cluster_val_loss[c])
                        w_fedAvg_best[c] = w_glob_fedAvg
                        cluster_model_max_iteration[c] = iteration

                    else:
                        cluster_counter[c] += 1

            if np.min(cluster_counter) >= patience:
                mylogger.info(f"Early stopping triggered in FL iteration {iteration}.")
                best_iteration = iteration
                break

        # Setting cluster models to best models found
        for c in range(args.clusters):
            if c in w_fedAvg_best:
                net_clusters[c].load_state_dict(w_fedAvg_best[c])

        # Initialize user result dictionary
        client_results = {idx: {} for idx in range(args.num_clients)}

        val_acc_locals, val_acc_mix, val_acc_ensemble = [], [], []
        val_acc_fedavg, val_acc_e2e = [], []
        val_acc_3, val_acc_rep, val_acc_repft, val_acc_ft = [], [], [], []
        val_acc_e2e_neighbour, val_acc_gateonly = [], []
        train_acc_ft, train_acc_locals = [], []
        acc_test_l, acc_test_m = [], []
        gate_values = []
        finetuned = []

        # TODO: in all of these cases we don't reuse `client`
        # TODO: Need to do for both clusters?

        if args.finetuning:
            mylogger.info("Starting finetuning")
            for idx in range(args.num_clients):

                client = ClientUpdate(args=args,
                                      train_set=dataset_train,
                                      val_set=dataset_test,
                                      idxs_train=dict_users[idx],
                                      idxs_val=dict_users_test[idx],
                                      parent_id=myid,
                                      client_id=idx)

                # finetune FedAvg for every client
                # TODO: Fine-tune all cluster models
                mylogger.debug(f"Finetuning for client {idx}")

                cluster_train_loss = []

                for c in range(args.clusters):
                    _, val_loss_fed = client.validate(
                        net=copy.deepcopy(net_clusters[c]).to(args. device),
                        train=True)
                    cluster_train_loss.append(val_loss_fed)

                # Returns all indicies
                c_indicies = np.where(
                    cluster_train_loss == np.min(cluster_train_loss))

                # Pick one on randoms
                c_idx = np.random.choice(c_indicies[0], 1)[0]

                # TODO: Remove magical constants
                wt, _, val_acc_finetuned, train_acc_finetuned, best_epoch = client.train_finetune(
                    net=copy.deepcopy(net_clusters[c_idx]).to(args.device),
                    n_epochs=args.loc_epochs,
                    learning_rate=args.ft_lr)

                client_results[idx].update(
                    {"finetuning": {
                        "train": train_acc_finetuned,
                        "validation": val_acc_finetuned,
                        "best_epoch": best_epoch
                    }})

                val_acc_ft.append(val_acc_finetuned)
                train_acc_ft.append(train_acc_finetuned)

                ft_net = copy.deepcopy(net_clusters[0])
                ft_net.load_state_dict(wt)
                finetuned.append(ft_net)

        # Evaluate on a smaller set of clients for speed
        evaluation_set = np.random.choice(
            range(args.eval_num_clients),
            args.eval_num_clients,
            replace=False)

        if args.train_local:
            mylogger.info("Starting training local models")
            for idx in evaluation_set:

                client = ClientUpdate(args=args,
                                      train_set=dataset_train,
                                      val_set=dataset_test,
                                      idxs_train=dict_users[idx],
                                      idxs_val=dict_users_test[idx],
                                      parent_id=myid,
                                      client_id=idx)

                # train local model
                # TODO: Remove magical constants
                mylogger.debug(f"Training local model for client {idx}")
                w_l, _, val_acc_l, train_acc_l, best_epoch = client.train_finetune(
                    net=net_locals[idx].to(args.device),
                    n_epochs=args.loc_epochs,
                    learning_rate=args.local_lr)

                client_results[idx].update(
                    {"local": {
                        "train": train_acc_l,
                        "validation": val_acc_l,
                        "best_epoch": best_epoch
                    }})

                net_locals[idx].load_state_dict(w_l)
                val_acc_locals.append(val_acc_l)
                train_acc_locals.append(train_acc_l)

        mylogger.info("Starting FL evaluation")
        # TODO: Evaluate each cluster, best one the client belonged to?

        cluster_use = [0 for x in range(args.clusters)]

        # Bootstrap
        for idx in evaluation_set:

            client = ClientUpdate(args=args,
                                  train_set=dataset_train,
                                  val_set=dataset_test,
                                  idxs_train=dict_users[idx],
                                  idxs_val=dict_users_test[idx],
                                  parent_id=myid,
                                  client_id=idx)

            cluster_train_loss = []

            for c in range(args.clusters):
                # TODO: evaluate FedAvg on validation dataset on all clusters.
                # Take max.
                _, train_loss_fed = client.validate(
                    net=net_clusters[c].to(args.device), train=True)
                cluster_train_loss.append(train_loss_fed)

            # Returns all indicies
            # np.nanmin is required in case any returned loss is nan
            c_indicies = np.where(
                cluster_train_loss == np.nanmin(cluster_train_loss))

            # Pick one on random if multiple
            try:
                c_idx = np.random.choice(c_indicies[0], 1)[0]
            except ValueError:
                c_idx = np.random.randint(args.clusters)

            cluster_use[c_idx] += 1

            # Evaluate on validation set
            cluster_val_acc, _ = client.validate(
                net=net_clusters[c_idx].to(args.device), train=False)

            mylogger.debug(f"Client {idx} cluster {c_idx}, accuracy {cluster_val_acc:.2f}")

            client_results[idx].update(
                {"fedavg": {
                    "train": np.nan,
                    "validation": np.max(cluster_val_acc),
                    "cluster": int(c_idx),
                    "iteration": cluster_model_max_iteration[c_idx]
                }})

            val_acc_fedavg.append(np.max(cluster_val_acc))

        for c in range(args.clusters):
            mylogger.debug(f"Cluster {c} model is used {cluster_use[c]} times.")
            if cluster_use[c] == 0:
                mylogger.warning(f"Cluster {c} model is never used.")

        if args.ensembles:

            mylogger.info("Starting Ensemble validation")
            for idx in evaluation_set:

                client = ClientUpdate(args=args,
                                      train_set=dataset_train,
                                      val_set=dataset_test,
                                      idxs_train=dict_users[idx],
                                      idxs_val=dict_users_test[idx],
                                      parent_id=myid,
                                      client_id=idx)

                # mylogger.debug(f"Validating ensembles for client {idx}")

                # TODO: Increases memory use
                nets = [copy.deepcopy(net_clusters[c]).to(args.device)
                        for c in range(args.clusters) if cluster_use[c] > 0]

                if args.train_local:
                    nets += [copy.deepcopy(net_locals[idx]).to(args.device)]

                # TODO: This now only works on Cifar 10 :)
                ensemble_model = MyEnsemble(nets)

                val_acc_ensemble_k, _ = client.validate(ensemble_model)

                mylogger.debug(f"Client {idx} ensemble accuracy {val_acc_ensemble_k:.2f}")

                client_results[idx].update(
                    {"ensemble": {
                        "train": np.nan,
                        "validation": val_acc_ensemble_k
                    }})

                val_acc_ensemble.append(val_acc_ensemble_k)

            val_acc_avg_ensemble = np.mean(val_acc_ensemble)
            mylogger.info(f"Client average ensemble accuracy {val_acc_avg_ensemble:.2f}")

        mylogger.info("Starting MoE trainings")

        # TODO: Add each cluster model
        # TODO: Save local models to disk and load them when needed to save memory
        # or at least make only the ones in the evaluation set take up memory on the GPU.
        # or move them to CPU when not needed on the GPU
        for idx in evaluation_set:

            mylogger.debug(f"Training mixtures for client {idx}")

            client = ClientUpdate(args=args,
                                  train_set=dataset_train,
                                  val_set=dataset_test,
                                  idxs_train=dict_users[idx],
                                  idxs_val=dict_users_test[idx],
                                  parent_id=myid,
                                  client_id=idx)

            # The mixture is trained over all (used) cluster models + the local model
            nets = [copy.deepcopy(net_clusters[c]).to(args.device)
                    for c in range(args.clusters) if cluster_use[c] > 0]

            if args.train_local:
                nets += [copy.deepcopy(net_locals[idx]).to(args.device)]

            # TODO: FIX FIX FIX
            # Ugly hack when number of models is not the same as number of clusters

            if args.dataset == "femnist":
                gates_e2e[idx] = GateCNNFEMNIST(
                    args=args, nomodels=len(nets)).to(args.device)
            else:
                gates_e2e[idx] = GateCNNLeaf(
                    args=args, nomodels=len(nets)).to(args.device)

            gates_e2e[idx].apply(weights_init)

            _, _, val_acc_e2e_k, gate_values, best_epoch = client.train_3(
                nets=nets,
                gate=copy.deepcopy(gates_e2e[idx]),
                train_gate_only=args.train_gate_only,
                n_epochs=args.moe_epochs,
                early_stop=True,
                learning_rate=args.moe_lr,
                weight_decay=args.gate_weight_decay)

            client_results[idx].update(
                {"mixtures": {
                    "train": np.nan,
                    "validation": val_acc_e2e_k,
                    "best_epoch": best_epoch,
                    "gate_values": np.nan
                }})

            val_acc_e2e.append(val_acc_e2e_k)

        val_acc_avg_e2e = np.mean(val_acc_e2e)
        mylogger.info(f"Client average MoE accuracy {val_acc_avg_e2e:.2f}")

        # Calculate validation and test accuracies
        if args.train_local:
            val_acc_avg_locals = np.mean(val_acc_locals)
            train_acc_avg_locals = np.mean(train_acc_locals)
        else:
            val_acc_avg_locals = np.nan
            train_acc_avg_locals = np.nan


        # val_acc_avg_e2e = np.nan

        # val_acc_avg_e2e_neighbour = sum(val_acc_e2e_neighbour) / len(val_acc_e2e_neighbour)
        val_acc_avg_e2e_neighbour = np.nan

        # val_acc_avg_3 = sum(val_acc_3) / len(val_acc_3)
        val_acc_avg_3 = np.nan

        # val_acc_avg_gateonly = sum(val_acc_gateonly) / len(val_acc_gateonly)
        val_acc_avg_gateonly = np.nan

        # val_acc_avg_rep = sum(val_acc_rep) / len(val_acc_rep)
        val_acc_avg_rep = np.nan

        # val_acc_avg_repft = sum(val_acc_repft) / len(val_acc_repft)
        val_acc_avg_repft = np.nan

        val_acc_avg_fedavg = np.mean(val_acc_fedavg)

        if args.finetuning:
            ft_val_acc = np.mean(val_acc_ft)
            ft_train_acc = np.mean(train_acc_ft)
        else:
            ft_val_acc = np.nan
            ft_train_acc = np.nan

        ft_test_acc = np.nan

        # TODO should follow the same naming scheme as the CSV file
        with open(f'save/{args.experiment}/{myid}_{run}_{args.clusters}_{filename}.json', 'w') as outfile:
            json.dump(client_results, outfile)

        # TODO: Make experiment directories
        with open(f'save/{args.experiment}/{myid}_{filename}.csv', 'a') as f1:
            f1.write('{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}'.format(
                myid,
                args.dataset, args.model, args.epochs, args.local_ep,
                args.num_clients, args.iid, args.p, args.opt, args.n_data,
                args.train_frac,
                args.train_gate_only, val_acc_avg_e2e,
                val_acc_avg_e2e_neighbour, val_acc_avg_locals,
                val_acc_avg_fedavg, ft_val_acc, val_acc_avg_3,
                val_acc_avg_rep, val_acc_avg_repft, val_acc_avg_ensemble,
                acc_test_mix, acc_test_locals, acc_test_fedavg, ft_test_acc,
                ft_train_acc, train_acc_avg_locals,
                val_acc_avg_gateonly, args.overlap, run, args.clusters,
                args.eps, args.explore_strategy, best_iteration))
            f1.write("\n")
        mylogger.info(f"Done: {val_acc_avg_fedavg}, {val_acc_avg_ensemble}, {val_acc_avg_e2e}, {val_acc_avg_locals}")

        for w in tb_writers:
            if w:
                w.close()

    return val_acc_avg_locals, val_acc_avg_fedavg, val_acc_avg_e2e


if __name__ == '__main__':
    args = args_parser()
    main(args)
    sys.exit(0)
