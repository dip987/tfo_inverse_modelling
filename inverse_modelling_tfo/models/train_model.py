from torch.nn import Linear, ReLU
from inverse_modelling_tfo.data import generate_data_loaders
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim


if __name__ == '__main__':
    params = {
        'batch_size': 100, 'shuffle': False, 'num_workers': 2
    }
    train_loader, test_loader = generate_data_loaders(params)

    model = nn.Sequential(
        nn.Linear(6, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
        nn.ReLU(),
        nn.Linear(3, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
        nn.Flatten()
    ).cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times
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
            if i % 20 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/20}')
                running_loss = 0.0

    print('Finished Training')


