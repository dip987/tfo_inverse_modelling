"""
A set of custom loss function meant to be used with the ModelTrainer Class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional
from altair import overload
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sympy import Li
import torch
from inverse_modelling_tfo.data.datasets import DATA_LOADER_LABEL_INDEX, DATA_LOADER_EXTRA_INDEX


class LossTracker:
    """
    A class to track the loss values during training. This class is meant to be used with the ModelTrainer class
    """

    def __init__(self, loss_names: List[str]):
        self.tracked_losses = loss_names
        # Set types for the dictionaries
        self.epoch_losses: Dict[str, List[float]]
        # Create the dictionaries
        self.epoch_losses = {loss_name: [] for loss_name in loss_names}  # Tracks the average loss for each epoch
        self.step_loss_sum = {loss_name: 0.0 for loss_name in loss_names}  # Tracks the loss for each step(in an epoch)
        self.steps_per_epoch_count = {
            loss_name: 0 for loss_name in loss_names
        }  # Tracks the number of steps in an epoch

    def step_update(self, loss_name: str, loss_value: float) -> None:
        """
        Update the losses for a single step within an epoch by appending the loss to the per_step_losses dictionary
        """
        if loss_name in self.step_loss_sum:
            self.steps_per_epoch_count[loss_name] += 1
            self.step_loss_sum[loss_name] += loss_value
        else:
            raise ValueError(f"Loss name '{loss_name}' not found in the LossTracker list!")

    def epoch_update(self) -> None:
        """
        Update the loss tracker for the current epoch. This is meant to be called at the end of each epoch.

        Averages out the losses from all the steps and places the average onto the epoch_losses list. Clears the
        per step losses for the next epoch
        """
        for loss_name in self.epoch_losses.keys():
            if self.steps_per_epoch_count[loss_name] != 0:
                self.epoch_losses[loss_name].append(
                    self.step_loss_sum[loss_name] / self.steps_per_epoch_count[loss_name]
                )
                self.step_loss_sum[loss_name] = 0.0  # Reset
                self.steps_per_epoch_count[loss_name] = 0  # Reset

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
            self.step_loss_sum[loss_name] = 0.0
            self.steps_per_epoch_count[loss_name] = 0


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
        self.train_loss_name = "train_loss" if name is None else f"{name}_train_loss"
        self.val_loss_name = "val_loss" if name is None else f"{name}_val_loss"

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
    def turn_on_tracking(self) -> None:
        """
        Turn on the loss tracking
        """

    @abstractmethod
    def turn_off_tracking(self) -> None:
        """
        Turn off the loss tracking
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

    def __init__(self, torch_loss_object, column_indices: Optional[List[int]] = None, name: Optional[str] = None):
        """
        :param torch_loss_object: An initialized torch loss object
        :param column_indices: The indices of the columns to be used for the loss calculation. If None, all columns are
        considered. Defaults to None
        """
        super().__init__(name)
        if name is None:
            self.loss_tracker = LossTracker(["train_loss", "val_loss"])
        else:
            self.loss_tracker = LossTracker([f"{name}_train_loss", f"{name}_val_loss"])
        self.loss_func = torch_loss_object
        self._tracking_on = True
        self.column_indices = column_indices

    def __call__(self, model_output, dataloader_data, trainer_mode):
        if self.column_indices is not None:
            loss = self.loss_func(
                model_output[:, self.column_indices], dataloader_data[DATA_LOADER_LABEL_INDEX][:, self.column_indices]
            )
        else:
            loss = self.loss_func(model_output, dataloader_data[DATA_LOADER_LABEL_INDEX])

        # Update internal loss tracker
        if self._tracking_on:
            loss_name = self.train_loss_name if trainer_mode == "train" else self.val_loss_name
            self.loss_tracker.step_update(loss_name, loss.item())
        return loss

    def turn_on_tracking(self) -> None:
        self._tracking_on = True

    def turn_off_tracking(self) -> None:
        self._tracking_on = False

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

    Always does mu_a1 - mu_a0 to get the delta mu_a
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
        self.mu_a1_scale = scaler_scale[mu_a1_index]

        # Initialize and Underlying MSE Loss
        self.loss_func = TorchLossWrapper(torch.nn.MSELoss(), name=name)
        self.loss_tracker = self.loss_func.loss_tracker  # Act as a wrapper around the underlying loss tracker

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        ground_truth_pulsation_ratios = dataloader_data[DATA_LOADER_EXTRA_INDEX][:, self.pulsation_ratio_indicies]
        # Unscale the model output
        unscaled_pathlength = model_output[:, self.pathlength_indicies] * self.pathlength_scale + self.pathlength_mean
        unscaled_mu_a0 = model_output[:, self.mu_a0_index] * self.mu_a0_scale + self.mu_a0_mean
        unscaled_mu_a1 = model_output[:, self.mu_a1_index] * self.mu_a1_scale + self.mu_a1_mean
        # Calculate the predicted pulsation ratios
        predicted_del_mu_a = unscaled_mu_a1 - unscaled_mu_a0
        predicted_del_mu_a = predicted_del_mu_a.unsqueeze(1)  # Convert to 2D tensor
        predicted_bl_term = predicted_del_mu_a * unscaled_pathlength
        # Calculate the loss
        loss = self.loss_func(predicted_bl_term, [0, ground_truth_pulsation_ratios], trainer_mode)
        return loss

    def __str__(self) -> str:
        return "Beer-Lamberts Law based Physics loss comparing the predicted pulsation ratio to the ground truth(using the pathlengths and mu_a values)"

    def loss_tracker_epoch_ended(self) -> None:
        self.loss_tracker.epoch_update()

    def turn_on_tracking(self) -> None:
        self.loss_func.turn_on_tracking()

    def turn_off_tracking(self) -> None:
        self.loss_func.turn_off_tracking()


class BLPathlengthLossDelta(LossFunction):
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
        1. Delta Mua (mu_a1 - mu_a0) and the Pathlengths (Both Scaled using a StandardScaler)

    Note: Need to pass in the model output scaler(Which includes both mu_a and pathlength scalers) to the loss function
    """

    def __init__(
        self,
        delta_mu_index: int,
        pathlength_indicies: List[int],
        pulsation_ratio_indicies: List[int],
        model_output_scaler: StandardScaler,
        device: torch.device = torch.device("cuda"),
        name: Optional[str] = None,
    ) -> None:
        """
        :param delta_mu_index: The index of the delta mu_a in model's predictions columns. (mu_a1 - mu_a0)
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

        self.delta_mu_index = delta_mu_index
        self.pathlength_indicies = pathlength_indicies
        self.pulsation_ratio_indicies = pulsation_ratio_indicies
        self.device = device

        # Convert the Scaler to device(cuda/cpu)
        scaler_mean = torch.tensor(model_output_scaler.mean_, device=device).float()
        scaler_scale = torch.tensor(model_output_scaler.scale_, device=device).float()
        self.pathlength_mean = torch.index_select(scaler_mean, 0, torch.tensor(pathlength_indicies).to(device))
        self.pathlength_scale = torch.index_select(scaler_scale, 0, torch.tensor(pathlength_indicies).to(device))
        self.delta_mu_mean = scaler_mean[delta_mu_index]
        self.delta_mu_scale = scaler_scale[delta_mu_index]

        # Initialize and Underlying MSE Loss
        self.loss_func = TorchLossWrapper(torch.nn.MSELoss(), name=name)
        self.loss_tracker = self.loss_func.loss_tracker  # Act as a wrapper around the underlying loss tracker

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        ground_truth_pulsation_ratios = dataloader_data[DATA_LOADER_EXTRA_INDEX][:, self.pulsation_ratio_indicies]
        # Unscale the model output
        unscaled_pathlength = model_output[:, self.pathlength_indicies] * self.pathlength_scale + self.pathlength_mean
        # Calculate the predicted pulsation ratios
        predicted_del_mu_a = model_output[:, self.delta_mu_index] * self.delta_mu_scale + self.delta_mu_mean
        predicted_del_mu_a = predicted_del_mu_a.unsqueeze(1)  # Convert to 2D tensor
        predicted_bl_term = predicted_del_mu_a * unscaled_pathlength
        # Calculate the loss
        loss = self.loss_func(predicted_bl_term, [0, ground_truth_pulsation_ratios], trainer_mode)
        return loss

    def __str__(self) -> str:
        return "Beer-Lamberts Law based Physics loss comparing the predicted pulsation ratio to the ground truth(using the pathlengths and mu_a values)"

    def loss_tracker_epoch_ended(self) -> None:
        self.loss_tracker.epoch_update()

    def turn_on_tracking(self) -> None:
        self.loss_func.turn_on_tracking()

    def turn_off_tracking(self) -> None:
        self.loss_func.turn_off_tracking()


class SumLoss(LossFunction):
    """
    Sum of two loss functions

    Special note: The name of the independent losses tracked inside each LossFunction needs to unique between all losses
    being summed

    Addional notes: The internal loss does not update per step. Rather per epoch. To get per step values, you need to
    look at the loss tracker for the individual loss_funcs/i.e. the constituents
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
        self.train_losses = list(filter(lambda x: "train_loss" in x, all_names))
        self.val_losses = list(filter(lambda x: "val_loss" in x, all_names))
        self.loss_tracker = LossTracker(all_names + [self.train_loss_name, self.val_loss_name])

    def __call__(self, model_output, dataloader_data, trainer_mode) -> torch.Tensor:
        # Calculate one loss to get the typing correct
        loss = torch.tensor(0.0, device=model_output.device, dtype=torch.float32)
        # loss = self.weights_tensor[0] * self.loss_funcs[0](model_output, dataloader_data, trainer_mode)
        # for loss_func, weight in zip(self.loss_funcs[1:], self.weights_tensor[1:]):
        for loss_func, weight in zip(self.loss_funcs, self.weights_tensor):
            extra_loss_term = weight * loss_func(model_output, dataloader_data, trainer_mode)
            loss = torch.add(loss, extra_loss_term)

        return loss

    def _merge_per_step_losses(self):
        """
        Merge the per step losses from all the constituent losses into a single dictionary
        """
        for index, loss_func in enumerate(self.loss_funcs):
            dictionary = loss_func.loss_tracker.step_loss_sum
            for loss_name, loss in dictionary.items():
                self.loss_tracker.step_update(loss_name, loss * self.weights_list[index])  # Weighted sum
        self.loss_tracker.step_update(
            self.train_loss_name, sum([self.loss_tracker.step_loss_sum[loss_name] for loss_name in self.train_losses])
        )
        self.loss_tracker.step_update(
            self.val_loss_name, sum([self.loss_tracker.step_loss_sum[loss_name] for loss_name in self.val_losses])
        )

    def __str__(self) -> str:
        individual_loss_descriptions = [str(loss_func) for loss_func in self.loss_funcs]
        individual_loss_descriptions = "\n".join(individual_loss_descriptions)
        return f"""Sum of multiple loss functions. 
        Constituent Losses: {[func.name for func in self.loss_funcs]}
        Weights: {self.weights_list}
        Individual Loss Func Description:
        {individual_loss_descriptions}
        """

    def loss_tracker_epoch_ended(self) -> None:
        # Merge the per step losses onto the underlying loss tracker
        self._merge_per_step_losses()

        # Update individual loss trackers
        for loss_func in self.loss_funcs:
            loss_func.loss_tracker_epoch_ended()

        # Update self
        self.loss_tracker.epoch_update()

    def turn_on_tracking(self) -> None:
        for loss_func in self.loss_funcs:
            loss_func.turn_on_tracking()

    def turn_off_tracking(self) -> None:
        for loss_func in self.loss_funcs:
            loss_func.turn_off_tracking()


class TorchLossWithChangingWeight(LossFunction):
    """
    Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings
    """

    def __init__(
        self,
        loss_func: LossFunction,
        start_weight: float,
        end_weight: float,
        epoch_count: int,
        start_delay: int = 0,
        name: Optional[str] = None,
    ):
        """
        Loss whose weight changes linearly as training goes on. This is useful for implementing schedules/annealings.
        If start delay is set to 0, the weight will start changing from the first epoch. Otherwise, it will hold the
        start value for the first start_delay epochs before starting to change.

        Args:
            loss_func: The underlying loss function
            start_weight: The weight at the start of the epochs
            end_weight: The weight at the end of the epochs
            epoch_count: The total number of epochs
            start_delay: The number of epochs to wait before starting the weight change. During these epochs, the
            start_weight will be used
            name: The name of the loss function

        Parameters:
            current_epoch: The current epoch number. Current epoch starts at 0 and goes up to
            (total_epochs + start_delay - 1). After that point, it stops changing until reset is called
            total_epochs: The total number of epochs as defined by the user for loss weight scheduling. This does not
            include the start_delay(explained below).
            start_delay: The number of epochs to wait before starting the weight change
            weights: The weights for each epoch
            loss_tracker: The loss tracker object
            _tracking_on: A flag to turn on/off tracking

        """
        super().__init__(name)
        self.loss_func = loss_func
        self.train_loss_name = "train_loss" if name is None else f"{name}_train_loss"
        self.val_loss_name = "val_loss" if name is None else f"{name}_val_loss"
        self._tracking_on = True
        self.loss_tracker = LossTracker([self.train_loss_name, self.val_loss_name])
        self.weights = torch.linspace(start_weight, end_weight, epoch_count)
        self.weights = torch.cat([start_weight * torch.ones(start_delay), self.weights])
        self.current_epoch = 0
        self.total_epochs = epoch_count
        self.start_delay = start_delay

    def __call__(self, model_output, dataloader_data, trainer_mode):
        loss = self.weights[self.current_epoch].item() * self.loss_func(model_output, dataloader_data, trainer_mode)
        # Update internal loss tracker
        if self._tracking_on:
            loss_name = self.train_loss_name if trainer_mode == "train" else self.val_loss_name
            self.loss_tracker.step_update(loss_name, loss.item())
        return loss

    def loss_tracker_epoch_ended(self) -> None:
        if self.current_epoch < self.total_epochs + self.start_delay - 1:
            self.current_epoch += 1
        self.loss_tracker.epoch_update()

    def turn_on_tracking(self) -> None:
        self._tracking_on = True

    def turn_off_tracking(self) -> None:
        self._tracking_on = False

    def __str__(self) -> str:
        return f"Loss Function with changing weight: {self.loss_func},\n Start Weight: {self.weights[0].item()}, End Weight: {self.weights[-1].item()}, Total Epochs: {self.total_epochs}, Start Delay: {self.start_delay}"

    def reset(self) -> None:
        self.current_epoch = 0
        self.loss_tracker = LossTracker([self.train_loss_name, self.val_loss_name])


class SumLossBalanced(SumLoss):
    """
    Tries to balances the losses by normalizing them to the same scale. This is useful when the losses have different
    scales and you want to balance them out.
    """

    def __init__(
        self,
        loss_funcs: List[LossFunction],
        averaging_window: int = 5,
        start_delay: int = 10,
        name: Optional[str] = None,
    ):
        super().__init__(loss_funcs, [1.0] * len(loss_funcs), name)
        self.average_window = averaging_window
        self.start_delay = start_delay

    def __call__(self, model_output, dataloader_data, trainer_mode):
        epochs_completed = len(self.loss_tracker.epoch_losses[self.train_loss_name])
        left = 0 if epochs_completed < self.average_window else epochs_completed - self.average_window
        loss = torch.tensor(0.0, device=model_output.device, dtype=torch.float32)
        avg = torch.tensor(0.0, device=model_output.device, dtype=torch.float32)
        if epochs_completed > self.start_delay:
            for loss_func in self.loss_funcs:
                loss_key = loss_func.train_loss_name if trainer_mode == "train" else loss_func.val_loss_name
                avg += torch.mean(torch.tensor(self.loss_tracker.epoch_losses[loss_key][left:]))
            avg /= len(self.loss_funcs)
        else:
            avg = 1.0

        for loss_func in self.loss_funcs:
            extra_loss_term = loss_func(model_output, dataloader_data, trainer_mode) / avg
            loss = torch.add(loss, extra_loss_term)

        return loss
