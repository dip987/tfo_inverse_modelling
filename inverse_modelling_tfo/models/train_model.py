from typing import Tuple, Optional, Dict, Callable, Type
from copy import deepcopy
from ray import train
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader


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

    def __init__(
        self, model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, epochs: int, criterion
    ):
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.epochs = epochs
        self.criterion = criterion
        self.train_loss = [-1]
        self.validation_loss = [-1]
        self.combined_loss = []
        self.reporting = False
        # Set Placeholder values during init, to be updated by the factory
        self.optimizer: Optimizer  # Initialized By Factory
        self.dataloader_gen_func_: Callable  # Initialized By factory
        self.dataloader_gen_kargs_: Dict  # Initialized By factory

    def set_optimizer(self, optimizer_class: Type, kwargs: Dict) -> None:
        """Change the current optimizer. Call this method before calling run to see the effects

        Args:
            optimizer_class (Type): Name of the optimizer class (NOT an Optimizer object. e.g.: SGD, Adam)
            kwargs (Dict): Key Word arguments to be passed into the optimizer
        """
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def _set_default_optimizer(self) -> None:
        """Sets the default optimizer"""
        self.optimizer = SGD(self.model.parameters(), lr=3e-4, momentum=0.9)

    def change_batch_size(self, batch_size: int) -> None:
        """Changes DataLoader batch size
        (Because of how PyTorch libraries are defined, changing batchsize requires creating a new DataLoader)
        """
        try:
            self.dataloader_gen_kargs_["data_loader_params"]["batch_size"] = batch_size
            self.train_loader, self.validation_loader = self.dataloader_gen_func_(**self.dataloader_gen_kargs_)
        except:
            raise KeyError("Incorrectly configured ModelTrainerFactory")

    def run(self):
        """Run Training and store results. Each Run resets all old results"""
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
                train.report(
                    {
                        "train_loss": self.train_loss[-1],
                        "val_loss": self.validation_loss[-1],
                        "combined_loss": self.combined_loss[-1],
                    }
                )

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

    def __init__(
        self,
        model_class: Type,
        model_gen_kargs: Dict,
        dataloader_gen_func: Callable,
        dataloader_gen_kargs: Dict,
        epochs: int,
        criterion,
    ):
        self.model_class = model_class
        self.model_gen_kargs = model_gen_kargs
        self.dataloader_gen_func = dataloader_gen_func
        self.dataloader_gen_kargs = dataloader_gen_kargs
        # Assert types (Because I don't know how to keep the inputs to the callable ambiguous, without using a
        # protocol/ too lazy to do that)
        self.train_loader: DataLoader
        self.validation_loader: DataLoader
        self.train_loader, self.validation_loader = dataloader_gen_func(**dataloader_gen_kargs)
        self.epochs = epochs
        self.criterion = criterion

    def create(self) -> ModelTrainer:
        """Creates a ModelTrainer based on the given blueprint"""
        model = self.model_class(**self.model_gen_kargs)
        trainer = ModelTrainer(model, self.train_loader, self.validation_loader, self.epochs, self.criterion)
        trainer.dataloader_gen_func_ = self.dataloader_gen_func
        # Make sure the individual models cannot change the original gen args
        trainer.dataloader_gen_kargs_ = deepcopy(self.dataloader_gen_kargs)
        return trainer
