from typing import Callable, Dict, Type
from copy import deepcopy

from torch.utils.data import DataLoader
from inverse_modelling_tfo.models.train_model import ModelTrainer
from inverse_modelling_tfo.models.loss_funcs import LossFunction


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
        criterion: LossFunction,
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
        trainer.loss_func.reset()   # Reset the loss tracker
        trainer.dataloader_gen_func_ = self.dataloader_gen_func
        # Make sure the individual models cannot change the original gen args
        trainer.dataloader_gen_kargs_ = deepcopy(self.dataloader_gen_kargs)
        return trainer
