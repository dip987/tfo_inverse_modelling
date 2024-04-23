"""
A set of custom loss function meant to be used with the ModelTrainer Class
"""

from abc import ABC, abstractmethod
from hmac import new
from typing import Dict, List, Optional
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
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
        self.tracked_losses = loss_names
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

    def epoch_update(self) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch.

        Averages out the losses from all the steps and places the average onto the epoch_losses list. Clears the
        per step losses for the next epoch
        """
        for loss_name in self.epoch_losses.keys():
            if len(self.per_step_losses[loss_name]) == 0:
                continue
            self.epoch_losses[loss_name].append(
                sum(self.per_step_losses[loss_name]) / len(self.per_step_losses[loss_name])
            )
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
            plt.legend()

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
        3. loss_tracker_epoch_update (optional)
        4. reset (optional)
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self.loss_tracker: LossTracker

    @abstractmethod
    def __call__(self, model_output, dataloader_data, trainer_mode: str) -> torch.Tensor:
        """
        Calculate & return the loss
        :param model_output: The output of the model
        :param dataloader_data: The data from the dataloader (The length of this depends on the DataLoader used)
        :param trainer_mode: The mode of the trainer (train/validate)

        Implementation Notes: When implementing this method, make sure to update the loss tracker using the
        loss_tracker_step_update method
        """

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the loss function
        """

    def loss_tracker_epoch_update(self) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch by the
        ModelTrainer class
        """
        self.loss_tracker.epoch_update()

    def reset(self) -> None:
        """
        Reset
        """
        self.loss_tracker = LossTracker(list(self.loss_tracker.epoch_losses.keys()))

    @abstractmethod
    def loss_tracker_epoch_ended(self) -> None:
        """
        Called at the end of the epoch to perform any necessary operations in the ModelTrainer
        """


class TorchLossWrapper(LossFunction):
    """
    A simple wrapper around torch.nn loss functions. This lets us seemlessly integrate torch loss functions with our own
    ModelTrainer class.

    By default, it trackes two losses:
        1.  train_loss
        2.  val_loss
    """

    def __init__(self, torch_loss_object, name: Optional[str] = None):
        """
        :param torch_loss_object: An initialized torch loss object
        """
        super().__init__(name)
        if name is None:
            self.loss_tracker = LossTracker(["train_loss", "val_loss"])
        else:
            self.loss_tracker = LossTracker([f"{name}_train_loss", f"{name}_val_loss"])
        self.loss_func = torch_loss_object

    def __call__(self, model_output, dataloader_data, trainer_mode):
        loss = self.loss_func(model_output, dataloader_data[DATA_LOADER_LABEL_INDEX])

        # Update internal loss tracker
        if trainer_mode == "train":
            loss_name = "train_loss"
        else:
            loss_name = "val_loss"
        if self.name is not None:
            loss_name = f"{self.name}_{loss_name}"
        self.loss_tracker.step_update(loss_name, loss.item())

        return loss

    def __str__(self) -> str:
        return f"Torch Loss Function: {self.loss_func}"

    def loss_tracker_epoch_ended(self) -> None:
        self.loss_tracker.epoch_update()


class BLPathlengthLoss(LossFunction):
    """
    Beer-Lamberts Law based loss that tries to equate the pathlength x del mu to the pulsation ratio.

    This loss assumes that the model predicts fetal mu_a at two different concentratio levels as well as the average
    pathlength for each detector. The loss function then calculates the predicted pulsation ratio using the predicted
    pathlength and mu_a difference. Then it compares it to the ground truth pulsation ratio. Minimizing this loss is
    equivalent to minimizing the error in the difference between these two terms.

    This loss also expects the dataloader to have 3 outputs,
        1. Model Input (could be anything, unrelated to the loss)
        2. Labels (Again could be anything, unrelated to the loss)
        3. Ground Truth Pulsation Ratios (Unscaled)

    The Model Predictions should include
        1. Mu A at two different levels & Pathlengths (Both Scaled using a StandardScaler)

    Note: Need to pass in the model output scaler(Which includes both mu_a and pathlength scalers) to the loss function
    """

    def __init__(
        self,
        mu_a0_index: int,
        mu_a1_index: int,
        pathlength_indicies: List[int],
        pulsation_ratio_indicies: List[int],
        model_output_scaler: StandardScaler,
        device: torch.device = torch.device("cuda"),
        name: Optional[str] = None,
    ) -> None:
        """
        :param mu_a0_index: The index of the first mu_a in model's predictions columns
        :param mu_a1_index: The index of the second mu_a in model's predictions columns
        :param pathlength_indicies: The indices of the pathlengths in the model's predictions columns
        :param pulsation_ratio_indicies: The pulsation ratio indices in the data loader's extra data. (Note: The
        pulsation ratios should be unscaled)
        :param model_output_scaler: The scaler object used to scale the model's output. This will be used to unscale the
        model's predictions to be able to compare them with the ground truth values
        :param device: The device to use for the calculations

        (Note: The pathlength_scaler should be an sklearn.preprocessing.StandardScaler object! But type enforcing is
        giving me errors on the private attribute types. It will throw an error if any other Scaler is used!)
        """
        super().__init__(name)
        # Ensure that pathlength_scaler is a StandardScaler object
        assert isinstance(model_output_scaler, StandardScaler), "pathlength_scaler should be a StandardScaler object!"
        # Ensure index lengths are correct
        assert len(pathlength_indicies) == len(pulsation_ratio_indicies), "Pathlength and Pulsation length Mismatch!"

        self.mu_a0_index = mu_a0_index
        self.mu_a1_index = mu_a1_index
        self.pathlength_indicies = pathlength_indicies
        self.pulsation_ratio_indicies = pulsation_ratio_indicies
        self.device = device

        # Convert the Scaler to device(cuda/cpu)
        scaler_mean = torch.tensor(model_output_scaler.mean_, device=device).float()
        scaler_scale = torch.tensor(model_output_scaler.scale_, device=device).float()
        self.pathlength_mean = torch.index_select(scaler_mean, 0, torch.tensor(pathlength_indicies).to(device))
        self.pathlength_scale = torch.index_select(scaler_scale, 0, torch.tensor(pathlength_indicies).to(device))
        self.mu_a0_mean = scaler_mean[mu_a0_index]
        self.mu_a0_scale = scaler_scale[mu_a0_index]
        self.mu_a1_mean = scaler_mean[mu_a1_index]
        # # Convert all to float32
        # self.pathlength_mean = self.pathlength_mean.float()
        # self.pathlength_scale = self.pathlength_scale.float()
        # self.mu_a0_mean = self.mu_a0_mean.float()
        # self.mu_a0_scale = self.mu_a0_scale.float()
        # self.mu_a1_mean = self.mu_a1_mean.float()

        # Initialize and Underlying MSE Loss
        self.loss_func = TorchLossWrapper(torch.nn.MSELoss(), name=name)
        self.loss_tracker = self.loss_func.loss_tracker  # Act as a wrapper around the underlying loss tracker

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        ground_truth_pulsation_ratios = dataloader_data[DATA_LOADER_EXTRA_INDEX][:, self.pulsation_ratio_indicies]
        # Unscale the model output
        unscaled_pathlength = model_output[:, self.pathlength_indicies] * self.pathlength_scale + self.pathlength_mean
        unscaled_mu_a0 = model_output[:, self.mu_a0_index] * self.mu_a0_scale + self.mu_a0_mean
        unscaled_mu_a1 = model_output[:, self.mu_a1_index] * self.mu_a0_scale + self.mu_a0_mean
        # Calculate the predicted pulsation ratios
        predicted_del_mu_a = unscaled_mu_a1 - unscaled_mu_a0
        predicted_del_mu_a = predicted_del_mu_a.unsqueeze(1)  # Convert to 2D tensor
        # TODO: FACTOR of 100!
        predicted_bl_term = predicted_del_mu_a * unscaled_pathlength / 100  # Somehow we ended up with a 100x factor

        # Calculate the loss
        loss = self.loss_func(predicted_bl_term, [0, ground_truth_pulsation_ratios], trainer_mode)
        return loss

    def __str__(self) -> str:
        return "Beer-Lamberts Law based Physics loss comparing the predicted pulsation ratio to the ground truth(using \
            the pathlengths and mu_a values)"

    def loss_tracker_epoch_ended(self) -> None:
        self.loss_tracker.epoch_update()


class SumLoss(LossFunction):
    """
    Sum of two loss functions

    Special note: The name of the independent losses tracked inside each LossFunction needs to unique between all losses
    being summed

    Addional notes: The internal loss does not update per step. Rather per epoch. To get per step values, you need to
    look at the loss tracker for the loss_funcs
    """

    def __init__(self, loss_funcs: List[LossFunction], weights: List[float], name: Optional[str] = None):
        super().__init__(name)
        # Check validitiy of the loss names
        all_names = []
        self.loss_directory = []  # Holds a list of losses per constituent loss func.
        for loss_func in loss_funcs:
            all_names = all_names + list(loss_func.loss_tracker.epoch_losses.keys())
            self.loss_directory.append(list(loss_func.loss_tracker.epoch_losses.keys()))
        # must be unique
        assert len(all_names) == len(set(all_names)), "Loss function names should be unique!"

        # Check validity of the weights
        assert len(loss_funcs) == len(weights), "Number of loss functions and weights should match!"

        self.loss_funcs = loss_funcs
        self.weights_tensor = torch.tensor(weights, dtype=torch.float32)
        self.weights_list = weights
        self.loss_tracker = LossTracker(all_names)

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        # Calculate one loss to get the typing correct
        loss = self.weights_tensor[0] * self.loss_funcs[0](model_output, dataloader_data, trainer_mode)

        for loss_func, weight in zip(self.loss_funcs[1:], self.weights_tensor[1:]):
            loss = torch.add(loss, weight * loss_func(model_output, dataloader_data, trainer_mode))
        return loss

    def _merge_per_step_losses(self):
        """
        Merge the per step losses from all the constituent losses into a single dictionary
        """
        merged_dict = {}
        for index, loss_func in enumerate(self.loss_funcs):
            dictionary = loss_func.loss_tracker.per_step_losses
            for key, value in dictionary.items():
                new_loss_list = [loss * self.weights_list[index] for loss in value]
                merged_dict[key] = new_loss_list
        self.loss_tracker.per_step_losses = merged_dict

    def __str__(self) -> str:
        return "Sum of multiple loss functions"

    def loss_tracker_epoch_ended(self) -> None:
        # Merge the per step losses onto the underlying loss tracker
        self._merge_per_step_losses()

        # Update individual loss trackers
        for loss_func in self.loss_funcs:
            loss_func.loss_tracker_epoch_ended()

        # Update self
        self.loss_tracker.epoch_update()
