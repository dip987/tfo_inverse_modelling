from inverse_modelling_tfo.data import generate_data_loaders
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import pandas as pd

def create_perceptron_model(node_counts: List[int] = [6, 6, 3, 2, 1]):
    """Create a Multi-Layer Fully-Connected Perceptron based on the array node counts. The first element is the
    number of inputs to the network, each consecutive number is the number of nodes(inputs) in each 
    hidden layers and the last element represents the number of outputs.

    Args:
        node_counts (List[int], optional): Number of nodes in each layer in the model. Defaults to [6, 6, 3, 2, 1].
    """
    layers = [nn.Linear(node_counts[0], node_counts[1])]
    for index, count in enumerate(node_counts[1:-1], start=1):
        layers.append(nn.ReLU())
        layers.append(nn.Linear(count, node_counts[index + 1]))
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def train_model(model: nn.Module, optimizer: optim.Optimizer, criterion, train_loader: DataLoader, validation_loader: DataLoader, 
                epochs: int = 3, gpu_to_use: int = 3) -> Tuple:
    """Convenience function to train model

    Args:
        model (nn.Module): The model to train. Note, the model weights will change after calling train_model. Model will also be shifted to CUDA
        optimizer (optim.Optimizer): Optimizer to use
        criterion (_type_): Loss criterion
        train_loader (DataLoader): Training DataLoader
        validation_loader (DataLoader): Validation DataLoader
        epochs (int, optional): How many epochs to train. Defaults to 3.
        gpu_to_use (int, optional): Which GPU to use. Defaults to 3 (For Rishad).

    Returns:
        Tuple: (Training losses, Validation losses) averages for each epoch 
    """
    # Set correct GPU
    torch.cuda.set_device(gpu_to_use)

    model = model.cuda()

    # Losses
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        # Training Loop
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # to CUDA
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        training_losses.append(running_loss / len(train_loader))



        # Validation Loop
        running_loss = 0.0
        for i, data in enumerate(validation_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # to CUDA
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()
        validation_losses.append(running_loss / len(validation_loader))

    return training_losses, validation_losses


if __name__ == '__main__':
    # Example Code
    params = {
        'batch_size': 500, 'shuffle': False, 'num_workers': 2
    }
    data = pd.read_pickle(r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl')
    train, val = generate_data_loaders(data, params, ['SDD', 'Uterus Thickness', 'Maternal Wall Thickness', 'Maternal Mu_a', 'Fetal Mu_a', 'Wave Int'], ['Intensity'])

    model = create_perceptron_model()
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    train_model(model, optimizer, criterion, train, val, epochs=2)
