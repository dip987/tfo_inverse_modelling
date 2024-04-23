import unittest
import pandas as pd
import numpy as np
import torch
from torch.nn import MSELoss
from sklearn.preprocessing import StandardScaler
from inverse_modelling_tfo.models.custom_models import PerceptronBD
from inverse_modelling_tfo.models.train_model import ModelTrainer
from inverse_modelling_tfo.data import generate_data_loaders
from inverse_modelling_tfo.models.loss_funcs import LossTracker, BLPathlengthLoss, SumLoss, TorchLossWrapper


class TestLostTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_tracker = LossTracker(["train_loss", "val_loss"])

    def test_initiation(self):
        self.loss_tracker.reset()
        self.assertEqual(self.loss_tracker.epoch_losses, {"train_loss": [], "val_loss": []})
        self.assertEqual(self.loss_tracker.per_step_losses, {"train_loss": [], "val_loss": []})

    def test_loss_tracker_step_update(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update("train_loss", 1)
        self.loss_tracker.step_update("train_loss", 2)
        self.loss_tracker.step_update("val_loss", 3)
        self.loss_tracker.step_update("val_loss", 4)
        self.assertEqual(self.loss_tracker.per_step_losses["train_loss"], [1, 2])
        self.assertEqual(self.loss_tracker.per_step_losses["val_loss"], [3, 4])

    def test_loss_averaging_on_epoch_update(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update("train_loss", 1)
        self.loss_tracker.step_update("train_loss", 2)
        self.loss_tracker.step_update("val_loss", 3)
        self.loss_tracker.step_update("val_loss", 4)
        self.loss_tracker.epoch_update()
        self.assertEqual(self.loss_tracker.epoch_losses["train_loss"], [1.5])
        self.assertEqual(self.loss_tracker.epoch_losses["val_loss"], [3.5])

    def test_multiple_epoch_updates_work(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update("train_loss", 1)
        self.loss_tracker.step_update("train_loss", 2)
        self.loss_tracker.epoch_update()
        self.loss_tracker.step_update("train_loss", 3)
        self.loss_tracker.step_update("train_loss", 4)
        self.loss_tracker.epoch_update()
        self.assertEqual(self.loss_tracker.epoch_losses["train_loss"], [1.5, 3.5])


class TestTorchLossWrapper(unittest.TestCase):
    def setUp(self) -> None:
        self.torch_loss_object = TorchLossWrapper(MSELoss())

        # Just need a dummy trainer object to test the loss function, does not do anything else
        dummy_model = PerceptronBD([1, 2, 1])
        test_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        train_loader, val_loader = generate_data_loaders(test_data, {}, ["x"], ["y"])
        self.dummy_trainer = ModelTrainer(dummy_model, train_loader, val_loader, 10, self.torch_loss_object)

    def test_call(self):
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
        loss = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
        self.assertEqual(loss.item(), 0.0)  # Model output same as labels

    def test_tracked_loss_is_float(self):
        self.torch_loss_object.loss_tracker.reset()
        self.dummy_trainer.mode = "train"
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
        _ = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
        loss = self.torch_loss_object.loss_tracker.per_step_losses["train_loss"]
        self.assertIsInstance(loss[0], float)

    def test_loss_tracker_epoch_ended_correct_length(self):
        test_length = 10
        self.torch_loss_object.loss_tracker.reset()
        self.dummy_trainer.mode = "train"
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
        for i in range(10):
            _ = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
            _ = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
            self.torch_loss_object.loss_tracker_epoch_ended()
        self.assertEqual(len(self.torch_loss_object.loss_tracker.epoch_losses["train_loss"]), test_length)

    def test_train_validate_modes_write_to_correct_loss(self):
        self.torch_loss_object.loss_tracker.reset()
        for i in range(2):
            self.dummy_trainer.mode = "train"
            model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
            dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
            _ = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
            self.assertEqual(self.torch_loss_object.loss_tracker.per_step_losses["train_loss"], [0.0])
            self.assertEqual(self.torch_loss_object.loss_tracker.per_step_losses["val_loss"], [])
            self.dummy_trainer.mode = "validate"
            _ = self.torch_loss_object(model_output, dataloader_data, self.dummy_trainer.mode)
            self.assertEqual(self.torch_loss_object.loss_tracker.per_step_losses["train_loss"], [0.0])
            self.assertEqual(self.torch_loss_object.loss_tracker.per_step_losses["val_loss"], [0.0])
            self.torch_loss_object.loss_tracker_epoch_ended()


class TestBLPathlengthLoss(unittest.TestCase):
    def setUp(self):
        self.dummy_scaler = StandardScaler()  # No scaling
        self.dummy_scaler.scale_ = np.array([1] * 4)
        self.dummy_scaler.mean_ = np.array([0] * 4)
        self.loss_func = BLPathlengthLoss(0, 1, [2, 3], [0, 1], self.dummy_scaler)

        # Just need a dummy trainer object to test the loss function, does not do anything else
        dummy_model = PerceptronBD([1, 2, 1])
        test_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        train_loader, val_loader = generate_data_loaders(test_data, {}, ["x"], ["y"])
        self.dummy_trainer = ModelTrainer(dummy_model, train_loader, val_loader, 10, self.loss_func)

    def test_call(self):
        model_output = torch.tensor([1.0, 1.0, 3.0, 4.0]).view(1, -1).cuda()  # mu0, mu1, pathlengthd1, pathlengthd2
        dataloader_data = [
            0,  # dummy
            0,  # dummy
            torch.tensor([1.0, 1.0]).view(1, -1).cuda(),
        ]  # ground truth pulsation ratios
        loss = self.loss_func(model_output, dataloader_data, self.dummy_trainer.mode)
        self.assertEqual(loss.item(), 1.0)  # error of 1.0 per detector -> averages to 0.5


class TestSumLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.loss1 = TorchLossWrapper(MSELoss(), "loss1")
        self.loss2 = TorchLossWrapper(MSELoss(), "loss2")
        self.sum_loss = SumLoss([self.loss1, self.loss2], [0.5, 0.5])

        # Just need a dummy trainer object to test the loss function, does not do anything else
        dummy_model = PerceptronBD([1, 2, 1])
        test_data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        train_loader, val_loader = generate_data_loaders(test_data, {}, ["x"], ["y"])
        self.dummy_trainer = ModelTrainer(dummy_model, train_loader, val_loader, 10, self.sum_loss)

    def test_call(self):
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        loss = self.sum_loss(model_output, dataloader_data, self.dummy_trainer.mode)
        # MSE -> 1^2 per loss -> 0.5 x 1 + 0.5 x 1 = 1
        self.assertEqual(loss.item(), 1.0)  # Model output same as labels

    def test_tracked_losses(self):
        expected_losses = self.loss1.loss_tracker.tracked_losses + self.loss2.loss_tracker.tracked_losses
        self.assertEqual(self.sum_loss.loss_tracker.tracked_losses, expected_losses)

    def test_losses_are_tracked_properly(self):
        self.sum_loss.loss_tracker.reset()
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        self.sum_loss(model_output, dataloader_data, self.dummy_trainer.mode)
        self.assertEqual(self.sum_loss.loss_tracker.per_step_losses["loss1_train_loss"], [1.0])
    
    # def test_weights_are_applied_properly(self):
    #     loss1 = TorchLossWrapper(MSELoss(), "loss1")
    #     loss2 = TorchLossWrapper(MSELoss(), "loss2")
    #     sum_loss = SumLoss([self.loss1, self.loss2], [1, 0.5])
    #     model_output = torch.tensor([1.0]).cuda()
    #     dataloader_data = [torch.tensor([2.0]).cuda()] * 2
    #     loss = sum_loss(model_output, dataloader_data, "train")
    #     self.assertEqual(sum_loss.loss_tracker.per_step_losses["loss1_train_loss"], [1.0])
    #     self.assertEqual(sum_loss.loss_tracker.per_step_losses["loss2_train_loss"], [0.5])



if __name__ == "__main__":
    unittest.main()
