from typing import List, Literal, Optional
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
    The first element is the number of inputs to the network, each consecutive number is the number of nodes(inputs)
    in each hidden layers and the last element represents the number of outputs. The number of layers is 1 less than
    the length of [node_counts]

    The input should be a 2D tensor with the shape (batch_size, feature_size). The output would also be a 2D tensor of
    shape (batch_size, output_feature_size), where output_feature_size is the last element of the [node_counts]
    """

    def __init__(self, node_counts: List[int], dropout_rates: Optional[List[float]] = None) -> None:
        """
        Args:
            node_counts: List[int] Node counts for the fully connected layers. The first element is the number of
            inputs to the network, each consecutive number is the number of nodes(inputs) in each hidden layers.
            dropout_rates: Optional[List[float]] Dropout rates for the fully connected layers. The length must be the 1
            less than the length of fc_node_counts. Set this to None to avoid Dropout Layers. (Analogous to setting
            dropout values to 0). Defaults to None / no dropout layer
        """
        # Sanity Check
        if dropout_rates is not None:
            assert len(node_counts) - 1 == len(dropout_rates), "length of dropout_rates must be 1 less than node counts"
        assert len(node_counts) > 1, "node_counts must have atleast 2 elements"
        super().__init__()
        self.layers: List[nn.Module]
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.BatchNorm1d(count))
            if dropout_rates is not None:
                self.layers.append(nn.Dropout1d(dropout_rates[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        # self.layers.append(nn.Tanh())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class FeatureResidual(torch.nn.Module):
    """
    The residual part of the FeatureResidualNetwork. This is not meant to be used as a stand-alone network. Use the
    FeatureResidualNetwork instead.
    """

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


class CNN1d(torch.nn.Module):
    """
    Create a fully convolutional neural network with 1D Convolutional Layers. Each convolution layer is followed by a
    bacthnorm, dropout and a ReLU activation function. The final layer has a Flatten layer to convert the 3D tensor to a
    2D tensor. The model is created using the nn.Sequential module. The model has the same amount of layers as the 
    length of the input_channels list. (All the input parameters should have the same length!)

    The input should be a 3D tensor with the shape (batch_size, channels, sequence_length). The output would be a 2D
    """

    def __init__(
        self,
        input_channels: List[int],
        output_channels: List[int],
        kernel_sizes: List[int],
        paddings: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dialations: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
    ):
        """
        Args:
            input_channels: List[int] Number of input channels for each of the convolution layers. The length should be
            the same as the number of layers
            output_channels: List[int] Number of output channels for each of the convolution layers. The length should
            be the same as the number of layers
            kernel_sizes: List[int] Kernel sizes for each of the convolution layers. The length should be the same as
            the number of layers
            paddings: List[int] Padding for each of the convolution layers. Defaults to None / padding = 1. If set to
            None, the padding is set to 1 for all layers. If set to a list, the length should be the same as the number
            of layers. Defaults to None / padding = 1
            strides: List[int] Strides for each of the convolution layers Defaults to None / stride = 1. If set to None,
            the stride is set to 1 for all layers. If set to a list, the length should be the same as the number of
            layers. Defaults to None / stride = 1
            dialations: List[int] Dialations for each of the convolution layers. Defaults to None / dialation = 1. If
            set to None, the dialation is set to 1 for all layers. If set to a list, the length should be the same as
            the number of layers. Defaults to None / dialation = 1.
            dropouts: Optional[List[float]] Dropout rates for each of the convolution layers. Set this to None to
            avoid Dropout Layers. (Analogous to setting dropout values to 0). If provided, The length must be equal to
            the number of layers. Defaults to None
        """
        # Sanity Check - Make sure all arguments are of the same length
        assert len(input_channels) == len(output_channels) == len(kernel_sizes), "All lists must be of the same length"
        if paddings is not None:
            assert len(paddings) == len(input_channels), "All lists must be of the same length"
        else:
            paddings = [1] * len(input_channels)
        if strides is not None:
            assert len(strides) == len(input_channels), "All lists must be of the same length"
        else:
            strides = [1] * len(input_channels)
        if dialations is not None:
            assert len(dialations) == len(input_channels), "All lists must be of the same length"
        else:
            dialations = [1] * len(input_channels)
        if dropouts is not None:
            assert len(dropouts) == len(input_channels), "All lists must be of the same length"

        super().__init__()
        self.layers: List[nn.Module]
        self.layers = [
            nn.Conv1d(input_channels[0], output_channels[0], kernel_sizes[0], strides[0], paddings[0], dialations[0])
        ]
        for index, count in enumerate(input_channels[1:], start=1):
            self.layers.append(nn.BatchNorm1d(count))
            if dropouts is not None:
                self.layers.append(nn.Dropout1d(dropouts[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv1d(
                    count,
                    output_channels[index],
                    kernel_sizes[index],
                    strides[index],
                    paddings[index],
                    dialations[index],
                )
            )
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class FC2CNN(torch.nn.Module):
    """
    A set of fully connected layers (with BatchNorm and Dropout) followed by a set of CNNs. The CNN part is modeled
    after the UNET architecture.

    The FC part consists of a set of fully connected layers with BatchNorm and Dropout with Relu. While the CNN part
    consists of a set of Conv2d layers with BatchNorm and Dropout with a relu.

    The input should be a 2D tensor with the shape (batch_size, feature_size). The output would also be a 2D tensor
    of shape (batch_size, output_feature_size), where output_feature_size is the last element of the [cnn_node_counts]
    """

    # TODO: There is a bug with odd numbered output layer count.
    def __init__(
        self,
        fc_node_counts: List[int],
        cnn_node_counts: List[int],
        kernel_sizes: List[int],
        fc_dropouts: Optional[List[float]] = None,
        cnn_dropouts: Optional[List[float]] = None,
        final_layer: Literal["sigmoid", "tanh", "none"] = "none"
    ):
        """
        Args:
            fc_node_counts: List[int] Node counts for the fully connected layers. The first element is the number of
            inputs to the network, each consecutive number is the number of nodes(inputs) in each hidden layers.
            cnn_node_counts: List[int] Output node counts for the CNN layers. The first element is the number of output
            layers for the first layer of CNN. It's input size is the same as the output size of the last FC layer. The
            padding is set accordingly to maintain the output size.
            kernel_sizes: List[int] Kernel sizes for each of the CNN layers. Must be the same length as cnn_node_counts
            fc_dropouts: List[float] Dropout rates for the fully connected layers. The length must be the 1 less than
            the length of fc_node_counts. Set this to None to avoid Dropout Layers. (Analogous to setting dropout values
            to 0). Defaults to None / no dropout layer
            cnn_dropouts: List[float] Dropout rates for the CNN layers. Can be set to None to avoid having a dropout
            layer altogether. If provided, The length must be the same length as cnn_node_counts. Defaults to None
            final_layer: Literal["sigmoid", "tanh", "none"]: The final layer of the network. Set to 'none' to have the 
            network end with a convnet. Otherwise, the last layer is appeneded after the final conv layer. 
            Defaults to "none"
        """
        # Sanity Check
        if fc_dropouts is not None:
            assert len(fc_node_counts) - 1 == len(fc_dropouts), "fc_dropouts must be 1 less than fc_node_counts"
        assert len(cnn_node_counts) == len(kernel_sizes), "kernel_sizes must be the same length as cnn_node_counts"
        if cnn_dropouts is not None:
            assert len(cnn_node_counts) == len(cnn_dropouts), "cnn_dropouts must be the same length as cnn_node_counts"

        # Initialize Variables
        super().__init__()
        self.fc = PerceptronBD(fc_node_counts)
        self.cnn_node_counts = torch.tensor(cnn_node_counts)
        self.kernel_sizes = torch.tensor(kernel_sizes)

        # Calculate the paddings for the CNN layers
        cnn_input_size = torch.tensor([fc_node_counts[-1]] + cnn_node_counts[:-1])
        paddings = (self.cnn_node_counts - cnn_input_size + self.kernel_sizes - 1) // 2

        self.cnn = CNN1d(
            input_channels=[1] * len(cnn_node_counts),
            output_channels=[1] * len(cnn_node_counts),
            kernel_sizes=kernel_sizes,
            paddings=paddings.tolist(),
            dropouts=cnn_dropouts,
        )
        final_layer_mapping = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "none": nn.Identity()}
        self.final_layer = final_layer_mapping[final_layer]
        
        self.layers = self.fc.layers + self.cnn.layers + [self.final_layer]

    def forward(self, x):
        return self.final_layer(self.cnn(self.fc(x).unsqueeze(1)))
