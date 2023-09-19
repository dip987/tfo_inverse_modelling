from typing import List, Tuple, Optional, Dict, Callable, Type
from copy import deepcopy
from ray import tune
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from .validation_methods import ValidationMethod
from sklearn.preprocessing import StandardScaler


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
                 criterion):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = None
        self.train_loss = [-1]
        self.validation_loss = [-1]
        self.combined_loss = []
        self.reporting = False
        self.dataloader_gen_func_ = None    # Initialized By factory
        self.dataloader_gen_kargs_ = None   # Initialized By factory


    def set_optimizer(self, optimizer_class: Type, kwargs: Dict) -> None:
        """Change the current optimizer. Call this method before calling run to see the effects

        Args:
            optimizer_class (Type): Name of the optimizer class (NOT an Optimizer object. e.g.: SGD, Adam)
            kwargs (Dict): Key Word arguments to be passed into the optimizer
        """
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def _set_default_optimizer(self) -> None:
        """Sets the default optimizer
        """
        self.optimizer = SGD(self.model.parameters(), lr=3e-4, momentum=0.9)
    
    def change_batch_size(self, batch_size: int) -> None:
        """Changes DataLoader batch size
        (Because of how PyTorch libraries are defined, changing batchsize requires creating a new DataLoader)
        """
        try:
            self.dataloader_gen_kargs_['data_loader_params']['batch_size'] = batch_size
            self.train_loader, self.validation_loader = self.dataloader_gen_func_(**self.dataloader_gen_kargs_)
        except:
            raise KeyError('Incorrectly configured ModelTrainerFactory')
        

    def run(self):
        """Run Training and store results. Each Run resets all old results
        """
        # Check for Optimizer
        if self.optimizer is None:
            self._set_default_optimizer()

        self.model = self.model.cuda()
        # Rest Losses
        self.train_loss = []
        self.validation_loss = []

        # Train Model
        self.model = self.model.train()
        for _ in range(self.epochs):  # loop over the dataset multiple times
            # Training Loop
            running_loss = 0.0
            for data in self.train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            self.train_loss.append(running_loss / len(self.train_loader))

            # Validation Loop
            self.model = self.model.eval()
            running_loss = 0.0
            for data in self.validation_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # to CUDA
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()
            self.validation_loss.append(running_loss / len(self.validation_loader))
            
            # Update Combined Loss
            self.combined_loss.append(self.validation_loss[-1] * self.train_loss[-1])

            # Reporting
            if self.reporting:
                tune.report(train_loss=self.train_loss[-1],
                            val_loss=self.validation_loss[-1],
                            combined_loss=self.combined_loss[-1])
    
    def __str__(self) -> str:
        return f"""
        Model Properties:
        {self.model}
        Optimizer Properties"
        {self.optimizer}
        DataLoader Params: 
            Batch Size: {self.dataloader_gen_kargs_['data_loader_params']['batch_size']}
            Validation Method: {self.dataloader_gen_kargs_['validation_method']}
        Loss:
            Train Loss: {self.train_loss[-1]}
            Val. Loss: {self.validation_loss[-1]}"""
        

class ModelTrainerFactory:
    """
    Contains the blueprint to create ModelTrainer(s). Call create() to get a new ModelTrainer

    ## Notes
    1. The train and val. dataloaders are created using dataloader_gen params during initialization. Be default, all 
    generated ModelTrainers have the same dataloader underneath to save memory. But that can be changed later on.
    
    2. Each call to create() creates a new model using the model_class and model_params.
    """

    def __init__(self, model_class: Type, model_gen_kargs: Dict, dataloader_gen_func: Callable,
                 dataloader_gen_kargs: Dict, epochs: int, criterion):
        self.model_class = model_class
        self.model_gen_kargs = model_gen_kargs
        self.dataloader_gen_func = dataloader_gen_func
        self.dataloader_gen_kargs = dataloader_gen_kargs
        self.train_loader, self.validation_loader = dataloader_gen_func(**dataloader_gen_kargs)
        self.epochs = epochs
        self.criterion = criterion

    def create(self) -> ModelTrainer:
        """Creates a ModelTrainer based on the given blueprint
        """
        model = self.model_class(**self.model_gen_kargs)
        trainer = ModelTrainer(model, self.train_loader, self.validation_loader, self.epochs, self.criterion)
        trainer.dataloader_gen_func_ = self.dataloader_gen_func
        # Make sure the individual models cannot change the original gen args
        trainer.dataloader_gen_kargs_ = deepcopy(self.dataloader_gen_kargs)
        return trainer


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


def train_model_wtih_reporting(model: nn.Module, optimizer: Optimizer, criterion,
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
