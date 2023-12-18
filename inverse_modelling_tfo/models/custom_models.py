from typing import List, Optional
from sqlalchemy import over
import torch.nn as nn
import torch.nn.functional as F
import torch


class SplitChannelCNN(nn.Module):
    """A N channel based CNN connected to a set of FC layers. The N input channels of the CNN each
    take 1/N-th of the inputs. (Say if [input_length] is 40, and N=2, 1st channel would get the 1st
    20 and the next 20 would be fed to the 2nd channel). The CNN Outputs onto [cnn_out_channel]
    number of output channels. (NOTE: cnn_out_channel has to be divisible by N. This is because
    the out channels are maintain the same split/separation as the input. In other words, for the
    above example, for [cnn_out_channel] = 4, the first 2 out channels would only get info from the
    first split channel)

    Afterwards, they are connected to a set of linear layers with activations. The output length for
    each of these linear layers are supplied using [linear_array]
    """

    # TODO: Currently only using a single conv layer. More might be necessary!

    def __init__(
        self,
        split_count: int = 2,
        input_length: int = 40,
        cnn_out_channel: int = 4,
        kernel_size: int = 5,
        linear_array: List[int] = [2, 1],
    ) -> None:
        super().__init__()

        split_error_msg = "cnn_out_channel has to be divisible by split_count"
        assert cnn_out_channel % split_count == 0, split_error_msg
        self.split_count = split_count
        self.inputs_per_split = input_length // split_count
        self.conv1 = nn.Conv1d(split_count, cnn_out_channel, kernel_size, groups=split_count)
        self.conv_output_length = cnn_out_channel * (self.inputs_per_split + 1 - kernel_size)
        self.linear_layers: List[nn.Module]
        self.linear_layers = [nn.Linear(self.conv_output_length, linear_array[0])]
        for index, count in enumerate(linear_array[0:-1]):
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(nn.Linear(count, linear_array[index + 1]))
        self.linear_layers.append(nn.Flatten())
        self.linear_network = nn.Sequential(*self.linear_layers)

    def forward(self, x):
        x = x.view(-1, self.split_count, self.inputs_per_split)
        x = F.relu(self.conv1(x))
        x = nn.Flatten()(x)
        x = self.linear_network(x)
        return x


class PerceptronReLU(nn.Module):
    """A Multi-Layer Fully-Connected Perceptron based on the array node counts.
    The first element is the number of inputs to the network, each consecutive number is the number
    of nodes(inputs) in each hidden layers and the last element represents the number of outputs.
    """

    def __init__(self, node_counts: List[int]) -> None:
        super().__init__()
        self.layers: List[nn.Module]
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class PerceptronBN(nn.Module):
    """A Multi-Layer Fully-Connected Perceptron based on the array node counts with Batch Normalization!
    The first element is the number of inputs to the network, each consecutive number is the number
    of nodes(inputs) in each hidden layers and the last element represents the number of outputs.
    """

    def __init__(self, node_counts: List[int]) -> None:
        super().__init__()
        self.layers: List[nn.Module]
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.BatchNorm1d(count))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class PerceptronDO(nn.Module):
    """A Multi-Layer Fully-Connected Perceptron based on the array node counts with DropOut!
    ## How to Use
    1. The first element is the number of inputs to the network, each consecutive number is the number
    of nodes(inputs) in each hidden layers and the last element represents the number of outputs.
    2. You can set the Dropout Rate between each layer using dropout_rates. This defaults to 0.5 for each layer. Make
    sure it's length is 2 less than [node_counts]
    3. Set the Dropout rate to 0.0 to effectively nullify it
    """

    def __init__(self, node_counts: List[int], dropout_rates: Optional[List[float]] = None) -> None:
        super().__init__()
        self.layers: List[nn.Module]
        if dropout_rates is None:
            linear_layer_count = len(node_counts) - 1
            dropout_layer_count = linear_layer_count - 1
            dropout_rates = [0.5] * dropout_layer_count
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.Dropout1d(dropout_rates[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class PerceptronBD(nn.Module):
    """A Multi-Layer Fully-Connected Perceptron based on the array node counts with Batch Normalization and DropOut!
    The first element is the number of inputs to the network, each consecutive number is the number
    of nodes(inputs) in each hidden layers and the last element represents the number of outputs.
    """

    def __init__(self, node_counts: List[int], dropout_rates: Optional[List[float]] = None) -> None:
        super().__init__()
        self.layers: List[nn.Module]
        if dropout_rates is None:
            linear_layer_count = len(node_counts) - 1
            dropout_layer_count = linear_layer_count - 1
            dropout_rates = [0.5] * dropout_layer_count
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.BatchNorm1d(count))
            self.layers.append(nn.Dropout1d(dropout_rates[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        # self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class FeatureResidual(torch.nn.Module):
    def __init__(self, lookup_table: torch.Tensor, lookup_key_indices: torch.Tensor, feature_indices: torch.Tensor):
        super().__init__()
        self.lookup_table = lookup_table
        self.lookup_key_indices = lookup_key_indices
        self.feature_indices = feature_indices

    def forward(self, predicted_key, features):
        # dist: distances between each batch(rows) against each key in the lookup table (columns)
        dist = torch.cdist(predicted_key, self.lookup_table[:, self.lookup_key_indices])
        # collapse columns to get the index of the closest key for each batch(row), 1D
        closest_point_index = torch.argmin(dist, dim=1)
        # How array indices work in torch/numpy: the result has the same shape as the indices. Techinically since I
        # want a 2D array, the indices should also be 2D, each element would then correspind to (x, y) to capture a
        # single element. But since I want the same index for each row, I can just use a 1D array and broadcast it.
        # Same for the columns, I want the same index for each column, so I can just use a 1D array and broadcast it.
        closest_point_features = self.lookup_table[closest_point_index.view(-1, 1), self.feature_indices.view(1, -1)]
        return features - closest_point_features

    def cpu(self):
        super().cpu()
        self.lookup_table = self.lookup_table.cpu()
        self.lookup_key_indices = self.lookup_key_indices.cpu()
        self.feature_indices = self.feature_indices.cpu()
        return self

    def cuda(self, device):
        super().cuda(device)
        self.lookup_table = self.lookup_table.cuda(device)
        self.lookup_key_indices = self.lookup_key_indices.cuda(device)
        self.feature_indices = self.feature_indices.cuda(device)
        return self

    def to(self, device):
        super().to(device)
        self.lookup_table = self.lookup_table.to(device)
        self.lookup_key_indices = self.lookup_key_indices.to(device)
        self.feature_indices = self.feature_indices.to(device)
        return self


class FeatureResidualNetwork(torch.nn.Module):
    """
    Generates residuals based on the predicted key and the lookup table, then uses this residual to make a final
    predictions.

    The network consists of two separate sequential NNs connected at the center by a residual generator. The first NN
    predicts a set of intermediate labels. The lookup table is used to fetch the features closes to this label
    combination. The residual generator then generates the difference between the input features and the fetched
    features. This residual is then passed on to the second NN which makes the final prediction.

    Arguments:
    --------
    node_count_left: List[int] Node counts for the first NN. Its connected to the input on one side and a residual
    generator on the other side. The last element must be the number of intermediate features which are passed on to the
    residual network to make predictions. This would predict our intermediate labels/keys
    dropout_rates_left: List[float] Dropout rates for the first NN
    node_count_right: List[int] Node counts for the second NN connected to the residual generator on one side and the
    output on the other side. The first element must be the number of intermediate labels/keys
    dropout_rates_right: List[float] Dropout rates for the second NN
    lookup_table: 2d torch.Tensor Lookup table containing a set of features and labels, this table is used to look up
    the intermediate feature values at the predicted key(by the first/left NN)
    lookup_key_indices: 1d torch.Tensor Indices of the lookup table that correspond to the keys/intermediate lables
    feature_indices: 1d torch.Tensor Indices of the lookup table that correspond to the input features
    """

    def __init__(
        self,
        node_count_left: List[int],
        dropout_rates_left: List[float],
        node_count_right: List[int],
        dropout_rates_right: List[float],
        lookup_table: torch.Tensor,
        lookup_key_indices: torch.Tensor,
        feature_indices: torch.Tensor,
    ):
        super().__init__()
        # Check validity
        assert (
            node_count_left[-1] == lookup_key_indices.shape[0]
        ), "Left Node List must end with the number of lookup keys"
        assert (
            node_count_right[0] == node_count_left[0]
        ), "First element of Right Node List must equal to the number of input features"

        # Create the NNs
        self.left_nn = PerceptronBD(node_count_left, dropout_rates_left)
        self.feature_residual = FeatureResidual(lookup_table, lookup_key_indices, feature_indices)
        self.right_nn = PerceptronBD(node_count_right, dropout_rates_right)

    def forward(self, x):
        path = self.left_nn(x)
        path = self.feature_residual(path, x)
        path = self.right_nn(path)
        return path

    def cpu(self):
        super().cpu()
        self.left_nn.cpu()
        self.feature_residual.cpu()
        self.right_nn.cpu()
        return self

    def cuda(self, device=None):
        super().cuda(device)
        self.left_nn.cuda(device)
        self.feature_residual.cuda(device)
        self.right_nn.cuda(device)
        return self

    def to(self, device):
        super().to(device)
        self.left_nn.to(device)
        self.feature_residual.to(device)
        self.right_nn.to(device)
        return self
