from ray import tune
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict, Callable
from .validation_methods import ValidationMethod


class ModelTrainer:
    """
    Convenience class to train and validate a model. Calling run() stores the results within the class.

    ## Initialization Notes
    1. To specify which GPU to use, set the environment variable. Example: os.environ["CUDA_VISIBLE_DEVICES"]="2"
    2. By default, trains using a SGD optimizer. You can change it using the function [.set_optimizer] before
    calling run()
    3. Similarly, any of the other properties can also be changed before calling run
    4. Turn on reporting when using with Ray Tune

    ## Results
    train_loss, validation_loss
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, epochs: int,
                 criterion_class):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.criterion_class = criterion_class
        self.optimizer = None
        self.train_loss = []
        self.validation_loss = []
        self.reporting = False

    def set_optimizer(self, optimizer_class, kwargs: Dict):
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def set_default_optimizer(self):
        self.optimizer = SGD(self.model.parameters(), lr=3e-4, momentum=0.9)

    def run(self):
        # Check for Optimizer
        if self.optimizer is None:
            self.set_default_optimizer()

        self.model = self.model.cuda()
        # Rest Losses
        self.train_loss = []
        self.validation_loader = []

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            # Training Loop
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion_class(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            self.train_loss.append(running_loss / len(self.train_loader))

            # Validation Loop
            running_loss = 0.0
            for i, data in enumerate(self.validation_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion_class(outputs, labels)

                # print statistics
                running_loss += loss.item()
            self.validation_loss.append(running_loss / len(self.validation_loader))

            if self.reporting:
                tune.report(train_loss=self.train_loss[-1], val_loss=self.validation_loss[-1],
                            combined_loss=self.train_loss[-1] * self.validation_loss[-1])


class ModelTrainerFactory:
    """
    Contains the blueprint to create ModelTrainer(s). Call create() to get a new ModelTrainer

    Each call to create() creates a new model using the model_class and model_params.
    The train and val. dataloaders are created using dataloader_gen params during initialization.
    """

    def __init__(self, model_class: nn.Module, model_gen_kargs: Dict, dataloader_gen_func: Callable,
                 dataloader_gen_kargs: Dict, epochs: int, criterion):
        self.model_class = model_class
        self.model_gen_kargs = model_gen_kargs
        self.train_loader, self.validation_loader = dataloader_gen_func(**dataloader_gen_kargs)
        self.epochs = epochs
        self.criterion = criterion

    def create(self) -> ModelTrainer:
        model = self.model_class(**self.model_gen_kargs)
        return ModelTrainer(model, self.train_loader, self.validation_loader, self.epochs, self.criterion)


def train_model(model: nn.Module, optimizer: Optimizer, criterion, train_loader: DataLoader,
                validation_loader: DataLoader, epochs: int = 3, gpu_to_use: int = 3) -> Tuple:
    """Convenience function to train model

    Args:
        model (nn.Module): The model to train. Note, the model weights will change after calling 
        train_model. Model will also be shifted to CUDA automatically
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


def train_model_wtih_reporting(model: nn.Module, optimizer: optim.Optimizer, criterion,
                               train_loader: DataLoader, validation_loader: DataLoader,
                               epochs: int = 3) -> Tuple:
    """Convenience function to train models with the ray tune workflow. Same function as the regular
    [train_model] excet it reports to ray tune on each turn. This allows for the optimizer to stop
    training early if the loss is too high.

    Reports 'training_loss' and 'val_loss' on each step.
    Note: With ray tune, torch cannot set the GPU id on its own. Tune has to do that

    Args:
        model (nn.Module): The model to train. Note, the model weights will change after calling 
        train_model. Model will also be shifted to CUDA automatically
        optimizer (optim.Optimizer): Optimizer to use
        criterion (_type_): Loss criterion
        train_loader (DataLoader): Training DataLoader
        validation_loader (DataLoader): Validation DataLoader
        epochs (int, optional): How many epochs to train. Defaults to 3.

    Returns:
        Tuple: (Training losses, Validation losses) averages for each epoch 
    """
    # Set correct GPU

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

        tune.report(train_loss=training_losses[-1], val_loss=validation_losses[-1],
                    combined_loss=training_losses[-1] * validation_losses[-1])

    return training_losses, validation_losses
