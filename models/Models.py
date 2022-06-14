import torch
from torch import nn
import torch.nn.functional as F


class MyEnsemble(nn.Module):
    """
    Variable size Ensemble model
    """

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        """
        """

        with torch.no_grad():
            out_list = torch.stack([model(x)[0] for model in self.models])

        return 0, self.activation(torch.sum(out_list, axis=0))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class MLP2(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP2, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class GateMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(GateMLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        return x


class CNNLeafFEMNIST(nn.Module):
    """
    Model from LEAF paper, but with dropout
    TODO: Implicit dimension choice for log_softmax has been deprecated
    """

    def __init__(self, args, model="local"):

        super().__init__()

        if model == "local":
            self.filters1 = args.localfilters1
            self.filters2 = args.localfilters2
            self.hiddenunits = args.localhiddenunits1
            self.dropout = args.localdropout
        elif model == "fl":
            self.filters1 = args.flfilters1
            self.filters2 = args.flfilters2
            self.hiddenunits = args.flhiddenunits1
            self.dropout = args.fldropout
        else:
            self.filters1 = 32
            self.filters2 = 64
            self.hiddenunits = 512
            self.dropout = 0.5

        self.conv1 = nn.Conv2d(1, self.filters1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.filters1, self.filters2, 5)
        self.fc1 = nn.Linear(self.filters2 * 4 * 4, self.hiddenunits)
        self.dropout = nn.Dropout(p = self.dropout)
        self.fc2 = nn.Linear(self.hiddenunits, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        """
        Forward pass
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filters2 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        out2 = self.activation(out1)
        return out1, out2


class GateCNNLeaf(nn.Module):

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

    def __init__(self, args, nomodels=None):
        super().__init__()

        self.gatehiddenunits1 = args.gatehiddenunits1
        self.gatefilters1 = args.gatefilters1
        self.gatefilters2 = args.gatefilters2
        self.gatedropout = args.gatedropout
        self.nomodels = args.clusters + 1

        if nomodels:
            self.nomodels = nomodels

        self.pool = nn.MaxPool2d(2, 2)

        if self.gatefilters2 > 0:
            self.conv1 = nn.Conv2d(args.channels, self.gatefilters1, 5)
            self.conv2_drop = nn.Dropout2d(p=self.gatedropout)
            self.conv2 = nn.Conv2d(self.gatefilters1, self.gatefilters2, 5)
            self.fc1 = nn.Linear(self.gatefilters2 * 5 * 5, self.gatehiddenunits1)
        else:
            self.conv1 = nn.Conv2d(args.channels, self.gatefilters1, 5)
            self.fc1 = nn.Linear(self.gatefilters1 * 14 * 14, self.gatehiddenunits1)

        self.dropout = nn.Dropout(p=self.gatedropout)

        if self.nomodels <= 2:
            self.fc2 = nn.Linear(self.gatehiddenunits1, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc2 = nn.Linear(self.gatehiddenunits1, self.nomodels)
            self.activation = nn.Softmax()

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        if self.gatefilters2 > 0:
            x = self.pool(F.relu(self.conv2_drop(self.conv2(x))))

        x = x.view(-1, self.num_flat_features(x)) # self.gatefilters2 * 5 * 5
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.activation(x)

        return x

class GateCNNFEMNIST(nn.Module):

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num

    def __init__(self, args, nomodels=None):
        super().__init__()

        self.gatehiddenunits1 = args.gatehiddenunits1
        self.gatefilters1 = args.gatefilters1
        self.gatefilters2 = args.gatefilters2
        self.gatedropout = args.gatedropout
        self.nomodels = args.clusters + 1

        if nomodels:
            self.nomodels = nomodels

        self.pool = nn.MaxPool2d(2, 2)

        if self.gatefilters2 > 0:
            self.conv1 = nn.Conv2d(args.channels, self.gatefilters1, 5)
            self.conv2_drop = nn.Dropout2d(p=self.gatedropout)
            self.conv2 = nn.Conv2d(self.gatefilters1, self.gatefilters2, 5)
            self.fc1 = nn.Linear(self.gatefilters2 * 4 * 4, self.gatehiddenunits1)
        else:
            self.conv1 = nn.Conv2d(args.channels, self.gatefilters1, 5)
            self.fc1 = nn.Linear(self.gatefilters1 * 12 * 12, self.gatehiddenunits1)

        self.dropout = nn.Dropout(p=self.gatedropout)

        if self.nomodels <= 2:
            self.fc2 = nn.Linear(self.gatehiddenunits1, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc2 = nn.Linear(self.gatehiddenunits1, self.nomodels)
            self.activation = nn.Softmax()

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))

        if self.gatefilters2 > 0:
            x = self.pool(F.relu(self.conv2_drop(self.conv2(x))))

        x = x.view(-1, self.num_flat_features(x)) # self.gatefilters2 * 5 * 5
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.activation(x)

        return x

class CNNIFCA(nn.Module):
    """
    Model from IFCA paper,
    TODO: Implicit dimension choice for log_softmax has been deprecated
    """

    def __init__(self, args, model="local"):

        super().__init__()

        if model == "local":
            self.filters1 = args.localfilters1
            self.filters2 = args.localfilters2
            self.hiddenunits1 = args.localhiddenunits1
            self.hiddenunits2 = args.localhiddenunits2
            self.dropout = args.localdropout
        elif model == "fl":
            self.filters1 = args.flfilters1
            self.filters2 = args.flfilters2
            self.hiddenunits1 = args.flhiddenunits1
            self.hiddenunits2 = args.flhiddenunits2
            self.dropout = args.fldropout
        else:
            self.filters1 = 64
            self.filters2 = 64
            self.hiddenunits1 = 384
            self.hiddenunits2 = 194
            self.dropout = 0.5

        self.conv1 = nn.Conv2d(3, self.filters1, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.norm1 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0)
        self.conv2 = nn.Conv2d(self.filters1, self.filters2, 5)
        self.norm2 = nn.LocalResponseNorm(4, alpha=0.001 / 9.0)

        self.fc1 = nn.Linear(self.filters2 * 5 * 5, self.hiddenunits1)
        self.fc2 = nn.Linear(self.hiddenunits1, self.hiddenunits2)

        self.dropout = nn.Dropout(p=self.dropout)
        self.fc3 = nn.Linear(self.hiddenunits2, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        """
        Forward pass
        """

        x = self.norm1(self.pool(F.relu(self.conv1(x))))
        x = self.pool(self.norm2(F.relu(self.conv2(x))))
        x = x.view(-1, self.filters2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out1 = F.relu(self.fc3(x))
        out2 = self.activation(out1)
        return out1, out2


class CNNLeaf(nn.Module):
    """
    Model from LEAF paper, but with dropout
    TODO: Implicit dimension choice for log_softmax has been deprecated
    """

    def __init__(self, args, model="local"):

        super().__init__()

        if model == "local":
            self.filters1 = args.localfilters1
            self.filters2 = args.localfilters2
            self.hiddenunits = args.localhiddenunits1
            self.dropout = args.localdropout
        elif model == "fl":
            self.filters1 = args.flfilters1
            self.filters2 = args.flfilters2
            self.hiddenunits = args.flhiddenunits1
            self.dropout = args.fldropout
        else:
            self.filters1 = 32
            self.filters2 = 64
            self.hiddenunits = 512
            self.dropout = 0.5

        self.conv1 = nn.Conv2d(3, self.filters1, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.filters1, self.filters2, 5)
        self.fc1 = nn.Linear(self.filters2 * 5 * 5, self.hiddenunits)
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc2 = nn.Linear(self.hiddenunits, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        """
        Forward pass
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filters2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        out2 = self.activation(out1)
        return out1, out2


class CNNCifar(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out1, out2


class GateCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nomodels = args.clusters + 1
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(120, 84)

        if self.nomodels <= 2:
            self.fc3 = nn.Linear(84, 1)
            self.activation = nn.Sigmoid()
        else:
            self.fc3 = nn.Linear(84, self.nomodels)
            self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class GateCNNSoftmax(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class CNNFashion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, args.num_classes)
        self.activation = nn.LogSoftmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out1 = F.relu(self.fc2(x))
        x = self.fc3(out1)
        out2 = self.activation(x)
        return out1, out2


class GateCNNFashion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x


class GateCNNFahsionSoftmax(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 4 * 4, 84)
        self.fc2 = nn.Linear(84, 42)
        self.fc3 = nn.Linear(42, 3)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x
