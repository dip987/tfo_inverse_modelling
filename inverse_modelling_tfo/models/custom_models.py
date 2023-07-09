from typing import List
from torch import nn
import torch.nn.functional as F


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

    def __init__(self, split_count: int = 2, input_length: int = 40, cnn_out_channel: int = 4,
                 kernel_size: int = 5, linear_array: List[int] = [2, 1]) -> None:
        super().__init__()
        split_error_msg = "cnn_out_channel has to be divisible by split_count"
        assert cnn_out_channel % split_count == 0, split_error_msg
        self.split_count = split_count
        self.inputs_per_split = input_length//split_count
        self.conv1 = nn.Conv1d(split_count, cnn_out_channel,
                               kernel_size, groups=split_count)
        self.conv_output_length = cnn_out_channel * \
            (self.inputs_per_split + 1 - kernel_size)
        self.linear_layers = [
            nn.Linear(self.conv_output_length, linear_array[0])]
        for index, count in enumerate(linear_array[0:-1]):
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(
                nn.Linear(count, linear_array[index + 1]))
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

    def __init__(self, node_counts) -> None:
        super().__init__()
        self.layers = [nn.Linear(node_counts[0], node_counts[1])]
        for index, count in enumerate(node_counts[1:-1], start=1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(count, node_counts[index + 1]))
        self.layers.append(nn.Flatten())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
