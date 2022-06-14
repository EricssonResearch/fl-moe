import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="global epochs")
    parser.add_argument('--num_clients', type=int, default=100, help="number of clients")
    parser.add_argument('--eval_num_clients', type=int, default=10, help="number of clients to evaluate on")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs for FL")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="FL learning rate")
    parser.add_argument('--ft_lr', type=float, default=1e-4, help="fine tuning learning rate")
    parser.add_argument('--moe_lr', type=float, default=1e-4, help="mixture of experts learning rate")
    parser.add_argument('--local_lr', type=float, default=1e-4, help="mixture of experts learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--eps', type=float, default=0.1, help="Epsilon")
    parser.add_argument('--n_data', type=float, default=500, help="datasize on each client")
    parser.add_argument('--train_frac', type=float, default=0.1, help="fraction of training data size")
    parser.add_argument('--n_data_test',type=float,default=100,help="test datasize on each client")
    parser.add_argument('--overlap', action='store_true', help='whether to allow label overlap between clients or not')
    # model argumentsiter
    parser.add_argument('--model', type=str, default='cnn', help='which model to use')

    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--channels', type=int, default=3, help="number of channels")
    # other arguments
    parser.add_argument('--explore_strategy', default='default', help='which exploration strategy to use')
    parser.add_argument('--clusters', type=int, default=1, help="Number of clusters")

    parser.add_argument('--train_gate_only', action='store_true', help='whether to train gate only or not')
    parser.add_argument('--gatehiddenunits1', type=int, default=512, help="Number of hidden units in fc layer of gate")
    parser.add_argument('--gatehiddenunits2', type=int, default=512, help="Number of hidden units in fc layer of gate")
    parser.add_argument('--gatefilters1', type=int, default=32, help="Number of hidden units in fc layer of gate")
    parser.add_argument('--gatefilters2', type=int, default=64, help="Number of hidden units in fc layer of gate")
    parser.add_argument('--gatefiltersize', type=int, default=5, help="Filter size in fc layer of gate")
    parser.add_argument('--gate_weight_decay', type=float, default=1e-4, help="MoE weight decay")
    parser.add_argument('--gatedropout', type=float, default=0.5, help="MoE dropout")

    parser.add_argument('--localhiddenunits1', type=int, default=512, help="Number of hidden units in fc layer 1 of local")
    parser.add_argument('--localhiddenunits2', type=int, default=512, help="Number of hidden units in fc layer 2 of local")
    parser.add_argument('--localfilters1', type=int, default=32, help="Number of hidden units in fc layer of local")
    parser.add_argument('--localfilters2', type=int, default=64, help="Number of hidden units in fc layer of local")
    parser.add_argument('--local_weight_decay', type=float, default=1e-4, help="Local weight decay")
    parser.add_argument('--localdropout', type=float, default=0.5, help="local dropout")

    parser.add_argument('--flhiddenunits1', type=int, default=512, help="Number of hidden units in fc layer 1 of FL")
    parser.add_argument('--flhiddenunits2', type=int, default=512, help="Number of hidden units in fc layer 2 of FL")
    parser.add_argument('--flfilters1', type=int, default=32, help="Number of hidden units in fc layer of local")
    parser.add_argument('--flfilters2', type=int, default=64, help="Number of hidden units in fc layer of local")
    parser.add_argument('--fl_weight_decay', type=float, default=1e-4, help="FL weight decay")
    parser.add_argument('--fldropout', type=float, default=0.5, help="local dropout")

    parser.add_argument('--ft_weight_decay', type=float, default=1e-4, help="Finetuning weight decay")

    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--runs', type=int, default=1, help='number of runs to do experiment')
    parser.add_argument('--opt', type=float, default=0.5, help='fraction of clients that opt-in (default: 0.5)')
    parser.add_argument('--p', type=float, default = 0.3, help='majority class percentage (default: 0.3)')
    parser.add_argument('--finetuning', action='store_true', help='whether to train finetuned models or not')
    parser.add_argument('--dataaugmentation', action='store_true', help='whether to train using data augmention or not')
    parser.add_argument('--ensembles', action='store_true', help='whether to train ensemble models or not')
    parser.add_argument('--tensorboard', action='store_true', help='output tensorboard logs')
    parser.add_argument('--train_local', action='store_true', help='train local models')

    # TODO: Rename
    parser.add_argument('--loc_epochs', type=int, default=200, help="number of iterations for local training")
    parser.add_argument('--moe_epochs', type=int, default=200, help="number of iterations for MOE training")

    # output arguments
    parser.add_argument('--filename', type=str, default='result', help='output filename')
    parser.add_argument('--experiment', type=str, default='result', help='output path')

    args = parser.parse_args()
    return args
