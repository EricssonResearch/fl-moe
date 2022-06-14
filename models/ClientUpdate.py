"""
Client Update
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils.util import get_logger
from torch.utils.tensorboard import SummaryWriter


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ClientUpdate(object):
    def __init__(self, args, train_set=None, val_set=None, idxs_train=None,
                 idxs_val=None, parent_id=None, client_id=None):

        if not parent_id:
            self.parent_id = "ClientUpdate"
            self.logger = get_logger("ClientUpdate")
        else:
            self.parent_id = parent_id
            self.logger = get_logger(parent_id)

        self.client_id = client_id

        self.args = args

        self.loss_func = nn.NLLLoss()

        self.lr = 1e-5
        if self.args.lr is not None:
            self.lr = self.args.lr

        self.writer = None
        # self.logger.debug(f"Learning rate: {self.lr}")
        if self.args.tensorboard and (self.parent_id is not None) and (self.client_id is not None):
            self.writer = SummaryWriter(
                f"save/{args.experiment}/{self.parent_id}/{self.client_id}")
        else:
            self.logger.warning(
                "Parent ID and Client ID not specified, won't write logs.")

        self.selected_clients = []
        self.train_set = DatasetSplit(train_set, idxs_train)
        #dataset_length = len(self.train_val_set)
        #self.train_set, _ = torch.utils.data.random_split(self.train_val_set,[round(args.train_frac*dataset_length),round((1-args.train_frac)*dataset_length)],generator=torch.Generator().manual_seed(23))

        self.ldr_train = DataLoader(
            self.train_set, batch_size=self.args.local_bs, shuffle=True)

        self.val_set = DatasetSplit(val_set, idxs_val)
        self.ldr_val = DataLoader(self.val_set, batch_size=1, shuffle=True)

    def __del__(self):
        if self.writer:
            self.writer.close()

    def lr_decay(self, global_step,
                 init_learning_rate=1e-3,
                 min_learning_rate=1e-5,
                 decay_rate=0.9999):

        lr = ((init_learning_rate - min_learning_rate) *
              pow(decay_rate, global_step) +
              min_learning_rate)
        return lr

    def train(self, net, n_epochs, validate=False, offset=0, weight_decay=0):
        net.train()
        # train and update
        lr0 = self.lr_decay(0, self.lr, self.lr / 100.0)

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr,
                                    weight_decay=weight_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda step: self.lr_decay(step, self.lr, self.lr / 100.0) / lr0)

        epoch_loss = np.inf

        for epoch in range(n_epochs):
            net.train()
            batch_loss = []

            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                _, log_probs = net(images)

                #log_probs = torch.log(log_probs)
                loss = self.loss_func(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss = sum(batch_loss) / len(batch_loss)

            if self.writer:
                self.writer.add_scalar(
                    'fl training loss', epoch_loss, epoch + offset)

        if validate:
            val_acc, val_loss = self.validate(net)
            if self.writer:
                self.writer.add_scalar(
                    'fl validation loss', val_loss, epoch + offset)

            return net.state_dict(), epoch_loss, val_acc, val_loss
            # print(val_acc)

        return net.state_dict(), epoch_loss


    def train_finetune(self, net, n_epochs, learning_rate, weight_decay=0):
        net.train()
        # train and update
        lr0 = self.lr_decay(0, learning_rate, learning_rate / 100.0)

        # TODO add parameter
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda step: self.lr_decay(step, learning_rate, learning_rate / 100.0) / lr0)

        patience = 10
        epoch_loss = []
        epoch_train_accuracy = []
        model_best = net.state_dict()
        train_acc_best = np.inf
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        best_epoch = 0

        for epoch in range(n_epochs):
            net.train()
            batch_loss = []
            correct = 0

            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                _, log_probs = net(images)

                #log_probs = torch.log(log_probs)
                loss = self.loss_func(log_probs, labels)

                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                _, predicted = torch.max(log_probs.data, 1)
                correct += (predicted == labels).sum().item()

            scheduler.step()
            train_accuracy = 100.00 * correct / len(self.ldr_train.dataset)
            epoch_train_accuracy.append(train_accuracy)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if self.writer:
                self.writer.add_scalar(
                    'local training loss', epoch_loss[-1], epoch)
                self.writer.add_scalar(
                    'local training accuracy', epoch_train_accuracy[-1], epoch)

            if epoch % 5 == 0:
                val_acc, val_loss = self.validate(net)
                #print(iter, val_loss)

                if self.writer:
                    self.writer.add_scalar(
                        'local learning rate', scheduler.get_last_lr()[0], epoch)
                    self.writer.add_scalar(
                        'local validation loss', val_loss, epoch)
                    self.writer.add_scalar(
                        'local validation accuracy', val_acc, epoch)

                if(val_loss < val_loss_best):
                    counter = 0
                    model_best = net.state_dict()
                    val_acc_best = val_acc
                    val_loss_best = val_loss
                    train_acc_best = train_accuracy
                    best_epoch = epoch
                    self.logger.debug(f"Finetuning Iter {epoch} | {val_acc_best:.2f}")
                else:
                    counter = counter + 1

                # Takes 50 iterations!
                if counter == patience:
                    return model_best, epoch_loss[-1], val_acc_best, train_acc_best, best_epoch

        return model_best, epoch_loss[-1], val_acc_best, train_acc_best, best_epoch

    def train_mix(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop, learning_rate):
        """
        TODO: Add training accuracy
        """

        net_local.train()
        net_global.train()
        gate.train()

        if(train_gate_only):
            optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW(list(net_local.parameters(
            )) + list(gate.parameters()) + list(net_global.parameters()), lr=learning_rate)

        patience = 10
        epoch_loss = []
        gate_best = gate.state_dict()
        val_acc_best = -np.inf
        val_loss_best = np.inf
        counter = 0
        gate_values_best = 0
        best_epoch = 0
        for epoch in range(n_epochs):

            net_local.train()
            net_global.train()
            gate.train()

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                net_local.zero_grad()
                net_global.zero_grad()
                gate.zero_grad()

                # TODO: Log gate weights
                gate_weight = gate(images)
                _, local_probs = net_local(images)
                _, global_probs = net_global(images)

                # TODO: update loss function
                log_probs = gate_weight * local_probs + \
                    (1 - gate_weight) * global_probs

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            this_epoch_loss = np.mean(batch_loss)

            if(early_stop):
                if(epoch % 5 == 0):
                    val_acc, val_loss, gate_values = self.validate_mix(
                        net_local, net_global, gate)

                    if(val_loss < val_loss_best):
                        counter = 0
                        gate_best = gate.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        gate_values_best = gate_values
                        best_epoch = epoch
                        self.logger.debug(f"MoE Iter {epoch} | {val_acc_best:.2f}")
                    else:
                        counter = counter + 1

                    if self.writer:
                        self.writer.add_scalar(
                            'moe validation loss', val_loss, epoch)

                        self.writer.add_scalar(
                            'moe validation accuracy', val_acc, epoch)

                    if counter == patience:
                        return gate_best, this_epoch_loss, \
                            val_acc_best, gate_values_best, best_epoch

        return gate_best, this_epoch_loss, val_acc_best, gate_values_best, best_epoch

    # def train_rep(self, net_local, net_global, gate, train_gate_only, n_epochs, early_stop):
    #     net_local.train()
    #     net_global.train()
    #     gate.train()

    #     if(train_gate_only):
    #         optimizer = torch.optim.Adam(gate.parameters(), lr=self.lr)
    #     else:
    #         optimizer = torch.optim.Adam(list(net_local.parameters(
    #         )) + list(gate.parameters()) + list(net_global.parameters()), lr=self.lr)

    #     patience = 10
    #     epoch_loss = []
    #     gate_best = gate.state_dict()
    #     val_acc_best = -np.inf
    #     val_loss_best = np.inf
    #     counter = 0
    #     gate_values_best = 0
    #     for iter in range(n_epochs):

    #         net_local.train()
    #         net_global.train()
    #         gate.train()

    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(
    #                 self.args.device), labels.to(self.args.device)

    #             net_local.zero_grad()
    #             net_global.zero_grad()
    #             gate.zero_grad()

    #             rep_local, _ = net_local(images)
    #             rep_global, _ = net_global(images)
    #             rep = torch.cat((rep_local, rep_global), 1)
    #             log_probs = gate(rep)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())

    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))

    #         if(early_stop):
    #             if(iter % 5 == 0):
    #                 val_acc, val_loss = self.validate_rep(
    #                     net_local, net_global, gate)
    #                 if(val_loss < val_loss_best):
    #                     counter = 0
    #                     gate_best = gate.state_dict()
    #                     val_acc_best = val_acc
    #                     val_loss_best = val_loss
    #                     print("Iter %d | %.2f" % (iter, val_acc_best))
    #                 else:
    #                     counter = counter + 1

    #                 if counter == patience:
    #                     return gate_best, epoch_loss[-1], val_acc_best, gate_values_best

    #     return gate_best, epoch_loss[-1], val_acc_best, gate_values_best


    def train_3(self, nets, gate, train_gate_only,
                n_epochs, early_stop, learning_rate, weight_decay=0):

        for net in nets + [gate]:
            net.train()

        params = list(gate.parameters())
        # TODO: Learning rate is different here!

        if not train_gate_only:
            params += sum([list(net.parameters()) for net in nets], [])

        #lr0 = self.lr_decay(0, learning_rate, learning_rate / 100.0)
        optimizer = torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                              lr_lambda=lambda step: self.lr_decay(step, learning_rate, learning_rate / 100.0) / lr0)

        patience = 20
        #epoch_loss = []

        gate_best = gate.state_dict()
        val_acc_best = 0
        val_loss_best = np.inf
        counter = 0
        gate_values_best = None
        best_epoch = 0

        for epoch in range(n_epochs):

            for net in nets + [gate]:
                net.train()

            batch_loss = []
            batch_accuracy = []
            correct = 0

            for _, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)

                for net in nets:
                    net.zero_grad()

                gate.zero_grad()

                gate_weight = gate(images)

                if len(nets) == 2:

                    _, global_probs = nets[0](images.float())
                    _, local_probs = nets[1](images.float())

                    log_probs = gate_weight *  torch.exp(local_probs) + \
                        (1 - gate_weight) *  torch.exp(global_probs)

                else:
                    log_probs = 0

                    for i in range(len(nets)):
                        # print(gate_weight.shape)
                        # print(gate_weight[:,i].shape)
                        _, net_probs = nets[i](images.float())
                        # print(net_probs.shape)
                        gate_weight_i = gate_weight[:, i].reshape(-1, 1)
                        log_probs += gate_weight_i * torch.exp(net_probs)
                        # print(log_probs.shape)

                _, predicted = torch.max(log_probs.data, 1)
                correct += (predicted == labels).sum().item()
                train_accuracy = 100.00 * correct / len(self.ldr_train.dataset)
                batch_accuracy.append(train_accuracy)

                loss = self.loss_func(torch.log(log_probs), labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            # scheduler.step()
            this_epoch_loss = np.mean(batch_loss)
            this_epoch_accuracy = np.mean(batch_accuracy)

            if self.writer:
                # self.writer.add_scalar(
                #         'moe learning rate', scheduler.get_last_lr()[0], epoch)
                self.writer.add_scalar(
                    'moe training loss', this_epoch_loss, epoch)
                self.writer.add_scalar(
                    'moe training accuracy', this_epoch_accuracy, epoch)
            #epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if(early_stop):
                if(epoch % 2 == 0):
                    val_acc, val_loss, gate_values = self.validate_3(
                        nets, gate)

                    if self.writer:
                        self.writer.add_scalar(
                            'moe validation loss', val_loss, epoch)
                        self.writer.add_scalar(
                            'moe validation accuracy', val_acc, epoch)

                    if(val_loss < val_loss_best):
                        counter = 0
                        gate_best = gate.state_dict()
                        val_acc_best = val_acc
                        val_loss_best = val_loss
                        gate_values_best = gate_values
                        best_epoch = epoch

                        self.logger.debug("Iter %d | %.2f" %
                                          (epoch, val_acc_best))
                    else:
                        counter = counter + 1

                    if counter == patience:
                        return gate_best, this_epoch_loss, val_acc_best, gate_values_best, best_epoch

        return gate_best, this_epoch_loss, val_acc_best, gate_values_best, best_epoch

    def validate(self, net, train=False):
        with torch.no_grad():
            net.eval()

            # validate
            loss = 0
            correct = 0

            if train:
                dataset = self.ldr_train
            else:
                dataset = self.ldr_val

            for _, (data, target) in enumerate(dataset):

                data, target = data.to(
                    self.args.device), target.to(self.args.device)
                _, log_probs = net(data)

                # sum up batch loss
                loss += self.loss_func(log_probs, target).item()

                # get the index of the max log-probability
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)
                                     ).long().cpu().sum()

            loss /= len(dataset.dataset)
            accuracy = 100.00 * correct / len(dataset.dataset)

        return accuracy.item(), loss

    def validate_mix(self, net_l, net_g, gate):
        with torch.no_grad():
            net_l.eval()
            net_g.eval()
            gate.eval()
            val_loss = 0
            correct = 0
            gate_values = np.array([])
            label_values = np.array([])
            for idx, (data, target) in enumerate(self.ldr_val):
                data, target = data.to(
                    self.args.device), target.to(self.args.device)

                # TODO: Log gate weights
                gate_weight = gate(data)

                gate_values = np.append(gate_values, gate_weight.item())
                label_values = np.append(label_values, target.item())

                _, local_probs = net_l(data)
                _, global_probs = net_g(data)

                log_probs = gate_weight * local_probs + \
                    (1 - gate_weight) * global_probs

                log_probs = torch.log(log_probs)
                val_loss += self.loss_func(log_probs, target).item()
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)
                                     ).long().cpu().sum()

            val_loss /= len(self.ldr_val.dataset)
            accuracy = 100.00 * correct / len(self.ldr_val.dataset)
            gl_values = np.concatenate(
                (gate_values.reshape(-1, 1), label_values.reshape(-1, 1)), axis=1)
            return accuracy.item(), val_loss, gl_values

    def validate_3(self, nets, gate):
        with torch.no_grad():
            for net in nets:
                net.eval()
            gate.eval()
            val_loss = 0
            correct = 0

            for _, (data, target) in enumerate(self.ldr_val):
                data, target = data.to(
                    self.args.device), target.to(self.args.device)

                gate_weight = gate(data)

                if len(nets) == 2:

                    _, global_probs = nets[0](data.float())
                    _, local_probs = nets[1](data.float())

                    log_probs = gate_weight * torch.exp(local_probs) + \
                        (1 - gate_weight) * torch.exp(global_probs)

                else:

                    log_probs = 0
                    for i, net in enumerate(nets):
                        _, net_probs = net(data.float())
                        log_probs += gate_weight[:, i] * torch.exp(net_probs)

                # gate_values = np.append(gate_values,gate_weight.item())
                # label_values = np.append(label_values,target.item())

                val_loss += self.loss_func(torch.log(log_probs), target).item()

                #_, y_pred = torch.max(log_probs.data, 1)
                #correct += (y_pred == target).sum().item()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)
                                     ).long().cpu().sum()

            val_loss /= len(self.ldr_val.dataset)
            accuracy = 100.00 * correct / len(self.ldr_val.dataset)
            #gl_values = np.concatenate((gate_values.reshape(-1,1), label_values.reshape(-1,1)),axis=1)
            return accuracy.item(), val_loss, gate_weight

    # def validate_rep(self, net_l, net_g, gate):
    #     with torch.no_grad():
    #         net_l.eval()
    #         net_g.eval()
    #         gate.eval()
    #         val_loss = 0
    #         correct = 0
    #         gate_values = np.array([])
    #         label_values = np.array([])
    #         for idx, (data, target) in enumerate(self.ldr_val):
    #             data, target = data.to(
    #                 self.args.device), target.to(self.args.device)

    #             #gate_values = np.append(gate_values,gate_weight.item())
    #             #label_values = np.append(label_values,target.item())

    #             rep_local, _ = net_l(data)
    #             rep_global, _ = net_g(data)
    #             rep = torch.cat((rep_local, rep_global), 1)
    #             log_probs = gate(rep)
    #             val_loss += self.loss_func(log_probs, target).item()
    #             y_pred = log_probs.data.max(1, keepdim=True)[1]
    #             correct += y_pred.eq(target.data.view_as(y_pred)
    #                                  ).long().cpu().sum()

    #         val_loss /= len(self.ldr_val.dataset)
    #         accuracy = 100.00 * correct / len(self.ldr_val.dataset)
    #         gl_values = np.concatenate(
    #             (gate_values.reshape(-1, 1), label_values.reshape(-1, 1)), axis=1)
    #         return accuracy.item(), val_loss
