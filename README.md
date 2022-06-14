# Adaptive Expert Models for Personalization in Federated Learning

This repo contains code for the paper Adaptive Expert Models for Personalization in Federated Learning to appear in [International Workshop on Trustworthy Federated Learning in Conjunction with IJCAI 2022 (FL-IJCAI'22)](https://federated-learning.org/fl-ijcai-2022/).

# Example

To run the code on `cifar100` with 50 clients, run the following line. Results will be saved in `/save/results_cifar10.csv`.

```
python main_fed.py --model 'cnn' --dataset 'cifar100' --n_data 100 --num_clients 50 --num_classes 100 --epochs 1500 --local_ep 3 --opt 0 --p 1.0 --gpu 0 --runs 1 --filename results_cifar10.csv
```

To see all options, run 
```
python main_fed.py -h
```

If you want to allow for overlap between class labels, pass the argument `--overlap`.

You can also specify a configuration file and iterate over a parameter by running 

```
python iterator.py --filename config_femnist.json
```

Note that for the FEMNIST dataset, you will have to generate the data by running the preprocessing script in the `leaf/data/femnist` subfolder. See [FEMNIST dataset](https://github.com/TalwalkarLab/leaf/tree/master/data/femnist)
for more information on this script.

It is possible to iterate over for example number of clusters, using configuration specified in a config file.

```
python iterator_clusters_old.py --explore_strategy eps --min_clusters 1 --max_clusters 6 --filename config_cifar10_small.json
```

# Results

See our paper.

# Docker

To run this code in Docker, use the following.

In subdirectory `docker` run `make debug` or in the root directory `docker run -it --rm -v `pwd`:/home/user/src martisak/fl-moe:latest bash`

You can build this image in the `docker` directory with `make build`.

# Cite

If you find this work useful, please cite us.

# Acknowledgements

The code developed in this repo was was adapted from [Specialized federated learning using a mixture of experts](https://github.com/edvinli/federated-learning-mixture) by [
Edvin Listo Zec](https://github.com/edvinli) which in turn was adapted from [Federated Learning ](https://github.com/shaoxiongji/federated-learning) by [Shaoxiong Ji](https://github.com/shaoxiongji).
