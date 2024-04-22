from typing import Dict, Callable, Type
from enum import Enum
from ray import train
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from inverse_modelling_tfo.data.data_loader import DATA_LOADER_INPUT_INDEX


class ModelTrainerMode(Enum):
    TRAIN = "train"
    VALIDATE = "validate"


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
        self.loss_func = criterion
        self.loss_tracker = self.loss_func.loss_tracker
        self.train_loss = []
        self.validation_loss = []
        self.combined_loss = []
        self.reporting = False
        self.mode = ModelTrainerMode.TRAIN
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
        # self.train_loss = []
        # self.validation_loss = []

        # Train Model
        self.model = self.model.train()
        for _ in range(self.epochs):  # loop over the dataset multiple times
            # Training Loop
            running_loss = 0.0
            self.mode = ModelTrainerMode.TRAIN
            for data in self.train_loader:
                # to CUDA
                inputs = data[DATA_LOADER_INPUT_INDEX].cuda()

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, data, self)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
            self.train_loss.append(running_loss / len(self.train_loader))
            self.loss_func.loss_tracker_epoch_ended(len(self.train_loader))

            # Validation Loop
            self.mode = ModelTrainerMode.VALIDATE
            self.model = self.model.eval()
            running_loss = 0.0
            for data in self.validation_loader:
                # get the inputs; data is a list of [inputs, labels]

                # to CUDA
                inputs = data[DATA_LOADER_INPUT_INDEX].cuda()

                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.loss_func(outputs, data, self)

                # print statistics
                running_loss += loss.item()
            self.validation_loss.append(running_loss / len(self.validation_loader))
            self.loss_func.loss_tracker_epoch_ended(len(self.validation_loader))

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
