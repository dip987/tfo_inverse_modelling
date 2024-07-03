"""
Custom Models
-------------
A set of extensions for the PyTorch nn.Module class. These let you quickly create custom models with a set of parameters
"""

from typing import List, Literal, Optional
from torch import nn
import torch


def he_initilization(module) -> None:
    """
    Initialize the weights of the module using the He initialization method. This is a common initialization method for
    ReLU based networks. The weights are initialized using a normal distribution with a mean of 0 and a standard
    deviation of sqrt(2 / n), where n is the number of input units in the weight tensor.

    Args:
        module: nn.Module The module whose weights need to be initialized

    Returns:
        None: The weights are initialized in-place
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


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
            dropout values to 0). Defaults to None / no dropout layer. You can also set the dropout rates to 0 to avoid
            dropout layers.
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
                if dropout_rates[index - 1] > 0:
                    self.layers.append(nn.Dropout1d(dropout_rates[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)
        self.apply(he_initilization)

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
        channel_counts: List[int],
        kernel_sizes: List[int],
        paddings: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dialations: Optional[List[int]] = None,
        dropouts: Optional[List[float]] = None,
        groups: Optional[List[int]] = None,
    ):
        """
        Args:
            channel_counts: List[int] Channel count throughout the CNN. The first n-1 elements are the input channels
            for each of the convolution layers. The last element is the output channel for the last layer. The length
            should be the 1 larger than the layer count
            kernel_sizes: List[int] Kernel sizes for each of the convolution layers. The length should be the same as
            the layer count
            paddings: List[int] Padding for each of the convolution layers. Defaults to None. If set to
            None, the padding is set to 0 for all layers. If set to a list, the length should be the same as the layer
            count. Defaults to None / padding = 1 for all layers
            strides: List[int] Strides for each of the convolution layers Defaults to None / stride = 1. If set to None,
            the stride is set to 1 for all layers. If set to a list, the length should be the same as the number of
            layers. Defaults to None / stride = 1
            dialations: List[int] Dialations for each of the convolution layers. Defaults to None / dialation = 1. If
            set to None, the dialation is set to 1 for all layers. If set to a list, the length should be the same as
            the number of layers. Defaults to None / dialation = 1.
            dropouts: Optional[List[float]] Dropout rates for each of the convolution layers. Set this to None to
            avoid Dropout Layers. (Analogous to setting dropout values to 0). If provided, The length must be equal to
            1 less than the number of layers(This is because the last layer does not have a dropout). Defaults to None
            groups: Optional[List[int]] Number of groups to split up the flow of datafor each of the convolution layers.
            Each group remains isolated from each other. Defaults to None / groups = 1 (A single group per layer. The
            entire layer is connected to the entire input channel)
        """
        # Sanity Check - Make sure all arguments are of the same length
        layer_count = len(channel_counts) - 1
        assert layer_count == len(kernel_sizes), "kernel_sizes length must be the same as layer count"
        if paddings is not None:
            assert layer_count == len(paddings), "paddings length must be the same as layer count"
        else:
            paddings = [0] * layer_count
        if strides is not None:
            assert layer_count == len(strides), "strides length must be the same as layer count"
        else:
            strides = [1] * layer_count
        if dialations is not None:
            assert layer_count == len(dialations), "dialations length must be the same as layer count"
        else:
            dialations = [1] * layer_count
        if groups is not None:
            assert layer_count == len(groups), "groups length must be the same as layer count"
        else:
            groups = [1] * layer_count
        if dropouts is not None:
            assert layer_count - 1 == len(dropouts), "dropouts length must be 1 less than as layer count"

        # Create the CNN Layers
        super().__init__()
        self.layers: List[nn.Module] = []
        for layer_index in range(layer_count):
            # Start by adding the BatchNorm, Dropout and ReLU for the previous layer
            self.layers.append(
                nn.Conv1d(
                    channel_counts[layer_index],
                    channel_counts[layer_index + 1],
                    kernel_sizes[layer_index],
                    strides[layer_index],
                    paddings[layer_index],
                    dialations[layer_index],
                    groups[layer_index],
                )
            )
            if layer_index != layer_count - 1:  # unless we are at the last layer, add the BatchNorm, Dropout, ReLU
                self.layers.append(nn.BatchNorm1d(channel_counts[layer_index + 1]))
                if dropouts is not None:
                    self.layers.append(nn.Dropout1d(dropouts[layer_index - 1]))
                self.layers.append(nn.ReLU())
            else:  # At the last layer, skip batch norp, relu, dropout and just add a flatten layer
                self.layers.append(nn.Flatten())

        self.model = nn.Sequential(*self.layers)
        self.apply(he_initilization)

    def forward(self, x):
        return self.model(x)


class FC2CNN(torch.nn.Module):
    """
    A set of fully connected layers followed by a set of CNNs (with BatchNorm + Dropout + ReLu on both). The CNN part is
    modeled after the UNET architecture.

    The purpose of this network is to take in a set of features and convert them into a set of spatially related
    features. The input should be a 2D tensor with the shape (batch_size, feature_size). The output would also be a 2D
    tensor of shape (batch_size, output_feature_size), where output_feature_size is the last element of the
    [cnn_node_counts]
    """

    # TODO: There is a bug with odd numbered output layer count.
    def __init__(
        self,
        fc_node_counts: List[int],
        cnn_node_counts: List[int],
        kernel_sizes: List[int],
        fc_dropouts: Optional[List[float]] = None,
        cnn_dropouts: Optional[List[float]] = None,
        final_layer: Literal["sigmoid", "tanh", "none"] = "none",
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
            channel_counts=[1] * (len(cnn_node_counts) + 1),
            kernel_sizes=kernel_sizes,
            paddings=paddings.tolist(),
            dropouts=cnn_dropouts,
        )
        final_layer_mapping = {"sigmoid": nn.Sigmoid(), "tanh": nn.Tanh(), "none": nn.Identity()}
        self.final_layer = final_layer_mapping[final_layer]

        self.layers = self.fc.layers + self.cnn.layers + [self.final_layer]

    def forward(self, x):
        return self.final_layer(self.cnn(self.fc(x).unsqueeze(1)))


class CNN2FC(torch.nn.Module):
    """
    A set of CNN layers followed by a set of fully connected layers (with BatchNorm + Dropout + ReLu on both).

    The purpose of this network is to take in a set of spatially related features and convert them into a set of
    independent features.

    The input should be a 3D tensor with the shape (batch_size, cnn_channel_counts[0], input_length). The output would
    be a 2D tensor of shape (batch_size, fc_output_node_counts[-1])
    """

    def __init__(
        self,
        input_length: int,
        cnn_channel_counts: List[int],
        cnn_kernel_sizes: List[int],
        fc_output_node_counts: List[int],
        cnn_paddings: Optional[List[int]] = None,
        cnn_strides: Optional[List[int]] = None,
        cnn_dialations: Optional[List[int]] = None,
        cnn_dropouts: Optional[List[float]] = None,
        cnn_groups: Optional[List[int]] = None,
        fc_dropout_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        self.cnn = CNN1d(
            cnn_channel_counts,
            cnn_kernel_sizes,
            cnn_paddings,
            cnn_strides,
            cnn_dialations,
            cnn_dropouts,
            cnn_groups,
        )
        # Determine the cnn output length
        self.cnn = self.cnn.eval()  # Set the model to evaluation mode first
        temp_tensor = torch.randn(1, cnn_channel_counts[0], input_length)  # Create a dummy tensor
        temp_tensor = self.cnn(temp_tensor)  # Pass it through the CNN, we should get a 2D tensor
        cnn_out_dim = temp_tensor.shape[1]  # Get the output dimension of the CNN using the dummy tensor
        self.cnn = self.cnn.train()  # Set the model back to training mode
        self.fc = PerceptronBD([cnn_out_dim] + fc_output_node_counts, fc_dropout_rates)
        self.layers = self.cnn.layers + self.fc.layers

    def forward(self, x):
        return self.fc(self.cnn(x))


class CNN2FC2dInput(CNN2FC):
    """
    A slightly modified version of CNN2FC that always takes in a single channel 2D input (batch size x feature length),
    instead of a 3D input. For more info check the CNN2FC class -
    """

    def __init__(
        self,
        input_length: int,
        cnn_out_channels: List[int],
        cnn_kernel_sizes: List[int],
        fc_output_node_counts: List[int],
        cnn_paddings: Optional[List[int]] = None,
        cnn_strides: Optional[List[int]] = None,
        cnn_dialations: Optional[List[int]] = None,
        cnn_dropouts: Optional[List[float]] = None,
        cnn_groups: Optional[List[int]] = None,
        fc_dropouts: Optional[List[float]] = None,
    ):
        """
        Args:
        input_length: int Length of the input features
        cnn_out_channels: List[int] Output channel counts for the CNN layers. The first element is the number of
        output layers for the first layer of CNN. It's input size is the same as the output size of the last FC layer.
        The padding is set accordingly to maintain the output size.
        cnn_kernel_sizes: List[int] Kernel sizes for each of the CNN layers. Must be the same length as cnn_out_channels
        fc_output_node_counts: List[int] Node counts for the fully connected layers. The first element is the number
        of inputs to the network, each consecutive number is the number of nodes(inputs) in each hidden layers.
        cnn_paddings: List[int] Padding for each of the CNN layers. Defaults to None / padding = 1. If set to None,
        the padding is set to 1 for all layers. If set to a list, the length should be the same as the layer count.
        Defaults to None / padding = 1 for all layers
        cnn_strides: List[int] Strides for each of the CNN layers Defaults to None / stride = 1. If set to None,
        the stride is set to 1 for all layers. If set to a list, the length should be the same as the number of
        layers. Defaults to None / stride = 1
        cnn_dialations: List[int] Dialations for each of the CNN layers. Defaults to None / dialation = 1. If set to
        None, the dialation is set to 1 for all layers. If set to a list, the length should be the same as the number
        of layers. Defaults to None / dialation = 1.
        cnn_dropouts: List[float] Dropout rates for each of the CNN layers. The length must be the 1 less than the
        length of fc_node_counts. Set this to None to avoid Dropout Layers. (Analogous to setting dropout values to 0).
        Defaults to None
        cnn_groups: List[int] Number of groups to split up the flow of datafor each of the CNN layers. Each group
        remains isolated from each other. Defaults to None / groups = 1 (A single group per layer. The entire layer is
        connected to the entire input channel)
        fc_dropouts: List[float] Dropout rates for the fully connected layers. The length must be the 1 less than
        the length of fc_node_counts. Set this to None to avoid Dropout Layers. (Analogous to setting dropout values
        to 0). Defaults to None
        """
        super().__init__(
            input_length,
            [1] + cnn_out_channels,
            cnn_kernel_sizes,
            fc_output_node_counts,
            cnn_paddings,
            cnn_strides,
            cnn_dialations,
            cnn_dropouts,
            cnn_groups,
            fc_dropouts,
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Convert the 2D input to a 3D input
        return self.fc(self.cnn(x))


class SplitChannelCNN(nn.Module):
    """A split-channel based CNN connected to a set of FC layers. This network splits the original input into separate
    split-channels where split-channel is connected to a separate filter bank and does not interact with each other
    before the FC layers. The idea is to extract features from different spatial regions. Once extracted, the features
    are concatenated and passed through the FC layers.

    This is a special case of the CNN2FC class defined above. The exception is that the input is 2D (Batch size x
    feature size) rather than 3D. The input is split into [split_count] separate channels before processing.

    The N input channels of the CNN each take 1/N-th of the complete_input_length. (Say if [complete_input_length] is 40
    and N=2, 1st channel would get the 1st 20 inputs and the next 20 inputs would be fed to the 2nd channel).
    """

    def __init__(
        self,
        complete_input_length: int,
        split_count: int,
        cnn_out_channels: List[int],
        cnn_kernel_sizes: List[int],
        fc_output_node_counts: List[int],
        cnn_paddings: Optional[List[int]] = None,
        cnn_strides: Optional[List[int]] = None,
        cnn_dialations: Optional[List[int]] = None,
        cnn_dropouts: Optional[List[float]] = None,
        fc_dropouts: Optional[List[float]] = None,
    ) -> None:
        # Sanity check
        layer_count = len(cnn_out_channels)
        assert complete_input_length % split_count == 0, "complete_input_length has to be divisible by split_count"
        self.input_length = complete_input_length // split_count
        # all elements of cnn_out_channls must be divisible by splot_count
        split_error_msg = "all elements of cnn_out_channel has to be divisible by split_count"
        assert all([x % split_count == 0 for x in cnn_out_channels]), split_error_msg
        assert len(cnn_kernel_sizes) == layer_count, "kernel_sizes length must be the same as cnn_out_channels length"
        super().__init__()
        self.network = CNN2FC(
            self.input_length,
            [split_count] + cnn_out_channels,
            cnn_kernel_sizes,
            fc_output_node_counts,
            cnn_paddings,
            cnn_strides,
            cnn_dialations,
            cnn_dropouts,
            [split_count] * len(cnn_out_channels),
            fc_dropouts,
        )
        self.split_count = split_count
        self.layers = self.network.layers

    def forward(self, x):
        x = x.view(-1, self.split_count, self.input_length)
        x = self.network(x)
        return x


class PMFEstimatorNet(nn.Module):
    def __init__(self, node_counts: List[int], dropout_rates: Optional[List[float]] = None):
        """
        A simple fully connected network that takes in a set of features and outputs a set of probabilities.
        The final layer is a sigmoid layer which gets normalized by the sum of all the outputs. All outputs are always
        positive.

        Args:
            node_counts: List[int] Node counts for the fully connected layers. The first element is the number of
            inputs to the network, each consecutive number is the number of nodes(inputs) in each hidden layers. The
            last element is the number of outputs
            dropout_rates: Optional[List[float]] Dropout rates for the fully connected layers. The length must be the 1
            less than the length of fc_node_counts. Set this to None to avoid Dropout Layers. (Analogous to setting
            dropout values to 0). Defaults to None / no dropout layer
        """
        super(PMFEstimatorNet, self).__init__()
        ## Sanity Check
        if dropout_rates is not None:
            assert len(node_counts) - 1 == len(dropout_rates), "length of dropout_rates must be 1 less than node counts"

        ## Create the Model Architecture
        self.layers: List[nn.Module] = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.BatchNorm1d(count))
            if dropout_rates is not None:
                self.layers.append(nn.Dropout1d(dropout_rates[index - 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Sigmoid())  # Finishes in a sigmoid layer

        ## Initialize the model
        self.model = nn.Sequential(*self.layers)
        self.apply(he_initilization)

    def forward(self, x):
        """
        Applies the model to the input tensor
        """
        x = self.model(x)
        x = x / torch.sum(x, dim=1, keepdim=True)
        return x


class DualPMFTMPNet(nn.Module):
    """
    A custom network that employs two separate PMF estimators for the pathlength distribution of two wavelenghts and
    then merges the output from those to to finally estimate saturation and concentration.
    """

    def __init__(
        self,
        pmf_nodes: List[int],
        terminal_fc_nodes: List[int],
        pmf_dropouts: Optional[List[float]] = None,
        terminal_fc_dropouts: Optional[List[float]] = None,
    ):
        """
        Args:
            pmf_nodes: List[int] Node counts for the fully connected layers of the PMF Estimator. The first element is
            the number of inputs to the network, each consecutive number is the number of nodes(inputs) in each hidden
            layers. The last element is the number of outputs
            terminal_fc_nodes: List[int] Node counts for the fully connected layers after the PMF Estimator. The first
            element is the number of inputs to the network, each consecutive number is the number of nodes(inputs) in each
            hidden layers. The last element is the number of outputs
            pmf_dropouts: Optional[List[float]] Dropout rates for the fully connected layers of the PMF Estimator. The
            length must be the 1 less than the length of fc_node_counts. Set this to None to avoid Dropout Layers.
            (Analogous to setting dropout values to 0). Defaults to None / no dropout layer
            terminal_fc_dropouts: List[float] Dropout rates for the fully connected layers after the PMF Estimator. The
            length must be the 1 less than the length of fc_node_counts. Set this to None to avoid Dropout Layers.
            (Analogous to setting dropout values to 0). Defaults to None / no dropout layer

        Model Input: Takes a 2D tensor, with the first half belonging to PMF1 and the second half belonging to PMF2
        Model Output: This model outputs a tuple of length 2: ([PMF1, PMF2], Output of Terminal FC)
        Custom Loss: This model requires a custom loss function that takes in the tuple output and the target values
        """
        ## Sanity Check
        if pmf_dropouts is not None:
            assert len(pmf_nodes) - 1 == len(pmf_dropouts), "length of pmf_dropouts must be 1 less than pmf_nodes"
        if terminal_fc_dropouts is not None:
            assert len(terminal_fc_nodes) == len(terminal_fc_dropouts), "terminal dropouts len must equal its nodes"

        ## Initialize the Model
        super(DualPMFTMPNet, self).__init__()
        self.pmf_estimator1 = PMFEstimatorNet(pmf_nodes, pmf_dropouts)
        self.pmf_estimator2 = PMFEstimatorNet(pmf_nodes, pmf_dropouts)
        terminal_fc_nodes = [pmf_nodes[-1] * 2] + terminal_fc_nodes
        self.terminal_fc = PerceptronBD(terminal_fc_nodes, terminal_fc_dropouts)

    def forward(self, x):
        pmf1 = self.pmf_estimator1(x[:, : x.shape[1] // 2])
        pmf2 = self.pmf_estimator2(x[:, x.shape[1] // 2 :])
        pmf = torch.cat([pmf1, pmf2], dim=1)
        return pmf, self.terminal_fc(pmf)


class SkipConnect(nn.Module):
    """
    A special variation of an MLP where certain inputs connect directly to a hidden layer.

    We deviced this network to incorporate depth data onto the MLP. Since the depth info is more impactful than the
    intensity, it makes sense to connect it deeper onto model path compared to the intensity data.
    """

    def __init__(
        self,
        node_counts: List[int],
        skip_connect_indices: List[int],
        skip_connect_layer_number: int,
        dropout_rates: Optional[List[float]] = None,
    ):
        """
        Args:
            node_counts: List[int] Node counts for the fully connected layers. [Input, Hidden1, Hidden2, ..., Output]
            The input node_count should not include the skip_connect_indices
            skip_connect_indices: List[int] Indices of the input features that should be connected directly to the
            skip_connect_layer_number layer.
            skip_connect_layer_number: int The layer number to which the skip_connect_indices should be connected to.
            0 conencts it to the input layer, 1 to the first hidden layer and so on.
            dropout_rates: Optional[List[float]] Dropout rates for the fully connected layers. The length must be the 1
            less than the length of fc_node_counts. Set this to None to avoid Dropout Layers.

        Design of the network:
        1. The network is divided into two parts: left and right. The left part is the part before the skip connection and
            the right part is the part after the skip connection.
        2. The left part is simple MLP that always ends in a linear layer. The skip connect indices are concatenated
            to the output of the left part.
        3. The right part is also a simple MLP that always starts with non-linear layers. The input to the right part
            is the output of the left part concatenated with the skip connect indices. This also ends in a linear layer.

        """
        ## Sanity Check
        if dropout_rates is not None:
            assert len(node_counts) - 1 == len(dropout_rates), "length of dropout_rates must be 1 less than node counts"
        else:
            dropout_rates = [0.0] * (len(node_counts) - 1)
        assert len(node_counts) > 1, "node_counts must have atleast 2 elements"
        assert skip_connect_layer_number < len(node_counts) - 1, "skip_connect_layer_number must be less than #layers"

        super().__init__()

        self.skip_indices = skip_connect_indices

        ## Create the Model Architecture
        # Break the network into two parts: before and after the skip connection
        self.node_list_left = node_counts[: skip_connect_layer_number + 1]
        dropout_rates_left = dropout_rates[:skip_connect_layer_number]
        self.node_list_right = node_counts[skip_connect_layer_number:]
        dropout_rates_right = dropout_rates[skip_connect_layer_number:]
        self.node_list_right[0] += len(skip_connect_indices)
        self.left_network: nn.Module
        self.righ_network: nn.Module

        # Define the left side of the network
        if skip_connect_layer_number == 0:
            self.left_network = nn.Identity()
        else:
            self.left_network_layers = [nn.Linear(self.node_list_left[0], self.node_list_left[1])]
            for index, count in enumerate(self.node_list_left[1:-1]):
                self.left_network_layers += [nn.BatchNorm1d(self.node_list_left[index + 1])]
                if dropout_rates_left[index] > 0:
                    self.left_network_layers += [nn.Dropout(dropout_rates_left[index])]
                self.left_network_layers += [nn.ReLU()]
                self.left_network_layers += [nn.Linear(count, self.node_list_left[index + 1])]
            self.left_network = nn.Sequential(*self.left_network_layers)

        # Define the right side of the network
        if skip_connect_layer_number == len(node_counts) - 1:
            self.right_network = nn.Identity()
        else:
            self.right_network_layers = []
            for index, count in enumerate(self.node_list_right[:-1]):
                self.right_network_layers.append(nn.BatchNorm1d(self.node_list_right[index]))
                if dropout_rates_right[index] > 0:
                    self.right_network_layers.append(nn.Dropout(dropout_rates_right[index]))
                self.right_network_layers.append(nn.ReLU())
                self.right_network_layers.append(nn.Linear(count, self.node_list_right[index + 1]))
            self.right_network_layers.append(nn.Flatten())
            self.right_network = nn.Sequential(*self.right_network_layers)
        # self.left_network.apply(he_initilization)
        # self.righ_network.apply(he_initilization)

    def forward(self, x):
        non_skip_indices = [i for i in range(x.shape[1]) if i not in self.skip_indices]
        skip_data = x[:, self.skip_indices]
        non_skip_data = x[:, non_skip_indices]
        left_output = self.left_network(non_skip_data)
        x = torch.cat([left_output, skip_data], dim=1)
        return self.right_network(x)
