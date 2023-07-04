from typing import List
from torch import nn
import torch.nn.functional as F


class TwoChannelCNN(nn.Module):
    """A 2 channel based CNN connected to a set of FC layers. The 2 input channels of the CNN each 
    take half of the inputs. (Say if [input_length] is 40, one channel would get the first 20 and 
    the next 20 would be fed to another channel). The CNN Outputs onto [cnn_out_channel] number of 
    output channels. (NOTE: cnn_out_channel has to be divisible by 2. The first half of out channle 
    only  see the first half of the inputs and the second half see the second half of the inputs)

    Afterwards, they are connected to a set of linear layers with activations. The output length for
    each of these linear layers are supploed using [linear_array]
    """

    def __init__(self, input_length: int, cnn_out_channel: int, kernel_size: int,
                 linear_array: List[int]) -> None:
        assert cnn_out_channel % 2 == 0, "cnn_out_channel has to be divisible by 2"
        super().__init__()
        self.split_point = input_length//2
        self.conv1 = nn.Conv1d(2, cnn_out_channel, kernel_size, groups=2)
        self.conv_output_length = cnn_out_channel * \
            (self.split_point + 1 - kernel_size)
        self.linear_layers = [
            nn.Linear(self.conv_output_length, linear_array[0])]
        for index, count in enumerate(linear_array[0:-1]):
            self.linear_layers.append(nn.ReLU())
            self.linear_layers.append(
                nn.Linear(count, linear_array[index + 1]))
        self.linear_layers.append(nn.Flatten())
        self.linear_network = nn.Sequential(*self.linear_layers)

    def forward(self, x):
        x = x.view(-1, 2, self.split_point)
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
