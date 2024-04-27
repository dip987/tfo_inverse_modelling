from typing import Dict, Type
from torch import nn
import torch
from torch.optim import SGD, Optimizer
from inverse_modelling_tfo.data.datasets import DATA_LOADER_INPUT_INDEX
from .DataLoaderGenerators import DataLoaderGenerator
from .validation_methods import ValidationMethod
from .loss_funcs import LossFunction
from torch.utils.data import DataLoader


class ModelTrainer:
    """
    Convenience class to train and validate a model. Call run() to train the model!

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
        self,
        model: nn.Module,
        dataloader_gen: DataLoaderGenerator,
        validation_method: ValidationMethod,
        loss_func: LossFunction,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.loss_func = loss_func
        # Call a reset on the loss tracker
        loss_func.loss_tracker.reset()
        self.dataloader_gen = dataloader_gen
        self.validation_method = validation_method
        self.train_loader: DataLoader
        self.validation_loader: DataLoader
        # Default optimizer
        self.optimizer: Optimizer
        self.optimizer = SGD(self.model.parameters(), lr=3e-4, momentum=0.9)
        self.device = device
        # Trackers
        self.reporting = False
        self.total_epochs = 0
        # Set initial mode to train
        self.mode = "train"

    def set_optimizer(self, optimizer_class: Type, kwargs: Dict) -> None:
        """Change the current optimizer. Call this method before calling run to see the effects

        Args:
            optimizer_class (Type): Name of the optimizer class (NOT an Optimizer object. e.g.: SGD, Adam)
            kwargs (Dict): Key Word arguments to be passed into the optimizer
        """
        self.optimizer = optimizer_class(self.model.parameters(), **kwargs)

    def set_batch_size(self, batch_size: int) -> None:
        """Changes DataLoader batch size
        (Because of how PyTorch libraries are defined, changing batchsize requires creating a new DataLoader)
        """
        self.dataloader_gen.change_batch_size(batch_size)
        self.train_loader, self.validation_loader = self.dataloader_gen.generate(self.validation_method)

    def run(self, epochs: int) -> None:
        """Run Training and store results. Each Run resets all old results"""
        self.model = self.model.to(self.device)
        self.train_loader, self.validation_loader = self.dataloader_gen.generate(self.validation_method)
        # Check device types for the model and the data loader
        # assert(self.model.device == self.train_loader.dataset.device)

        # Rest Losses
        # self.train_loss = []
        # self.validation_loss = []

        # Train Model
        for _ in range(epochs):  # loop over the dataset multiple times
            # Training Loop
            self.mode = "train"
            self.model = self.model.train()
            for data in self.train_loader:
                inputs = data[DATA_LOADER_INPUT_INDEX]

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, data, self.mode)
                loss.backward()
                self.optimizer.step()

            # Validation Loop
            self.mode = "validate"
            self.model = self.model.eval()
            for data in self.validation_loader:
                inputs = data[DATA_LOADER_INPUT_INDEX]

                with torch.no_grad():
                    outputs = self.model(inputs)
                    loss = self.loss_func(outputs, data, self.mode)

            self.loss_func.loss_tracker_epoch_ended()
            # Reporting
            if self.reporting:
                pass
                # TODO: Implement this part if needed in the future
        self.total_epochs += epochs

    def __str__(self) -> str:
        return f"""
        Model Properties:
        {self.model}
        Data Loader Properties:
        {self.dataloader_gen}
        Validation Method:
        {self.validation_method}
        Loss Function:
        {self.loss_func}
        Optimizer Properties":
        {self.optimizer}
        """
