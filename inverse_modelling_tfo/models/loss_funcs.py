"""
A set of custom loss function meant to be used with the ModelTrainer Class
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from inverse_modelling_tfo.models.train_model import ModelTrainer, ModelTrainerMode
from inverse_modelling_tfo.data.data_loader import (
    DATA_LOADER_LABEL_INDEX,
    DATA_LOADER_INPUT_INDEX,
    DATA_LOADER_EXTRA_INDEX,
)


class LossTracker:
    """
    A class to track the loss values during training. This class is meant to be used with the ModelTrainer class
    """

    def __init__(self, loss_names: List[str]):
        # Set types for the dictionaries
        self.epoch_losses: Dict[str, List[float]]
        self.per_step_losses: Dict[str, List[float]]
        # Create the dictionaries
        self.epoch_losses = {loss_name: [] for loss_name in loss_names}  # Tracks the average loss for each epoch
        self.per_step_losses = {loss_name: [] for loss_name in loss_names}  # Tracks the loss for each step(in an epoch)

    def step_update(self, loss_name: str, loss_value: float) -> None:
        """
        Update the losses for a single step within an epoch by appending the loss to the per_step_losses dictionary
        """
        if loss_name in self.per_step_losses:
            self.per_step_losses[loss_name].append(loss_value)
        else:
            raise ValueError(f"Loss name '{loss_name}' not found in the LossTracker list!")

    def epoch_update(self, epoch_data_points_length: int) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch
        """
        for loss_name in self.epoch_losses.keys():
            self.epoch_losses[loss_name].append(sum(self.per_step_losses[loss_name]) / epoch_data_points_length)
            self.per_step_losses[loss_name] = []

    def plot_losses(self) -> None:
        """
        Plot the losses on the current axes
        """
        if len(self.epoch_losses) == 0:
            print("No losses to plot!")
            return
        for loss_name, loss_values in self.epoch_losses.items():
            plt.plot(loss_values, label=loss_name)

    def reset(self):
        """
        Clears out all saved losses
        """
        for loss_name in self.epoch_losses.keys():
            self.epoch_losses[loss_name] = []
            self.per_step_losses[loss_name] = []


class LossFunction(ABC):
    """
    Base abstract class for all loss functions. All loss functions must inherit from this class and implement the
    following methods
        1. __call__
        2. __str__
        3. loss_tracker_step_update
        4. loss_tracker_epoch_update (optional)
        5. reset (optional)
    """

    def __init__(self):
        self.loss_tracker: LossTracker

    @abstractmethod
    def __call__(self, model_output, dataloader_data, trainer: ModelTrainer) -> torch.Tensor:
        """
        Calculate & return the loss
        :param model_output: The output of the model
        :param dataloader_data: The data from the dataloader (The length of this depends on the DataLoader used)
        :param trainer: The ModelTrainer object

        Implementation Notes: When implementing this method, make sure to update the loss tracker using the
        loss_tracker_step_update method
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the loss function
        """

    @abstractmethod
    def loss_tracker_step_update(self, *args):
        """
        Update the loss tracker with the current loss values. This should be called at the end of each __call__
        """

    def loss_tracker_epoch_update(self, epoch_data_points_length: int) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch by the
        ModelTrainer class
        """
        self.loss_tracker.epoch_update(epoch_data_points_length)

    def reset(self) -> None:
        """
        Reset the loss tracker
        """
        self.loss_tracker = LossTracker(list(self.loss_tracker.epoch_losses.keys()))


class TorchLossWrapper(LossFunction):
    """
    A simple wrapper around torch.nn loss functions. This lets us seemlessly integrate torch loss functions with our own
    ModelTrainer class.

    By default, it trackes three losses:
        1.  train_loss
        2.  val_loss
    """

    def __init__(self, torch_loss_object):
        """
        :param torch_loss_object: An initialized torch loss object
        """
        super().__init__()
        self.loss_tracker = LossTracker(["train_loss", "val_loss", "combined_loss"])
        self.loss_func = torch_loss_object

    def __call__(self, model_output, dataloader_data, trainer: ModelTrainer):
        loss = self.loss_func(model_output, dataloader_data[DATA_LOADER_LABEL_INDEX])
        if trainer.mode == ModelTrainerMode.TRAIN:
            self.loss_tracker.step_update("train_loss", loss.item())
        else:
            self.loss_tracker.step_update("val_loss", loss.item())
        return loss

    def __str__(self) -> str:
        return f"Torch Loss Function: {self.loss_func}"

    def loss_tracker_step_update(self, loss_name: str, loss_value: float):
        """
        Update the loss tracker with the current loss values
        """
        self.loss_tracker.step_update(loss_name, loss_value)


class BLPathlengthLoss(LossFunction):
    """
    Beer-Lamberts Law based loss that tries to equate the pathlength x del mu to the pulsation ratio.

    This loss assumes that the model predicts fetal mu_a at two different concentratio levels as well as the average
    pathlength for each detector. The loss function then calculates the predicted pulsation ratio using the predicted
    pathlength and mu_a difference. Then it compares it to the ground truth pulsation ratio. Minimizing this loss is
    equivalent to minimizing the error in the difference between these two terms.

    This loss also expects the dataloader to have 3 outputs, where the extra data should contain the ground truth
    pulsation ratios. The pulsation ratios should be unscaled.

    The loss also assumes that the pathlengths are scaled using a StandardScaler object. This is required to unscale the
    pathlengths from the model's predictions onto the original scale.
    """

    def __init__(
        self,
        mu_a0_index: int,
        mu_a1_index: int,
        pathlength_indicies: List[int],
        pulsation_ratio_indicies: List[int],
        pathlength_scaler,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """
        :param mu_a0_index: The index of the first mu_a in model's predictions columns
        :param mu_a1_index: The index of the second mu_a in model's predictions columns
        :param pathlength_indicies: The indices of the pathlengths in the model's predictions columns
        :param pulsation_ratio_indicies: The pulsation ratio indices in the data loader's extra data. (Note: The
        pulsation ratios should be unscaled)
        :param pathlength_scaler: The StandardScaler object used to scale the pathlengths. This is required to unscale
        the pathlengths from the model's predictions onto the original scale.
        (Note: The pathlength_scaler should be an sklearn.preprocessing.StandardScaler object! But type enforcing is
        giving me errors on the private attribute types. It will throw an error if any other Scaler is used!)
        :param device: The device to use for the calculations

        """
        super().__init__()
        # Ensure that pathlength_scaler is a StandardScaler object
        assert isinstance(pathlength_scaler, StandardScaler), "pathlength_scaler should be a StandardScaler object!"

        # Convert the internal variables to cuda
        self.pathlength_scaler_mean = torch.tensor(pathlength_scaler.mean_).to(device)
        self.pathlength_scaler_scale = torch.tensor(pathlength_scaler.scale_).to(device)

    def forward(self, prediction, targets):
        pass
        # Physics Term
        # scaled_mua0 = targets[:, 0].reshape(-1, 1) * self.label_scaler_scale[0] + self.label_scaler_mean[0]
        # scaled_mua1 = targets[:, 1].reshape(-1, 1) * self.label_scaler_scale[1] + self.label_scaler_mean[1]
        # predicted_del_mu_a = scaled_mua1 - scaled_mua0  # Mua1 - Mua0
        # predicted_avg_L = prediction[:, 2:] * self.label_scaler_scale[2:] + self.label_scaler_mean[2:]
        # predicted_bl_term = predicted_del_mu_a * predicted_avg_L / 100  # Somehow we ended up with a factor of 100 here
        # # TODO: Probably figure out how that came to be!
        # target_pulsation_ratios = targets[:, self.output_labels_len :]
        # target_pulsation_ratios = self.pathlength_scaler_scale * target_pulsation_ratios + self.pathlength_scaler_mean
        # # Assuming the pulsation ratios are the same as the BL terms
        # bl_loss = self.label_loss(target_pulsation_ratios, predicted_bl_term)

        # # Label Loss - How good is the model at predicting the given labels
        # target_labels = targets[:, : self.output_labels_len]
        # label_loss = self.label_loss(prediction, target_labels)

        # # Total Loss
        # total_loss = self.label_loss_weight * label_loss + self.bl_loss_weight * bl_loss
        # return total_loss
