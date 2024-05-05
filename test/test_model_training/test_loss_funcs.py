import unittest
import numpy as np
import torch
from torch.nn import MSELoss
from sklearn.preprocessing import StandardScaler
from inverse_modelling_tfo.model_training.loss_funcs import (
    LossTracker,
    BLPathlengthLoss,
    SumLoss,
    TorchLossWrapper,
    TorchLossWithChangingWeight,
)
# TODO: Some of these tests are outdated and need to be updated


class TestLostTracker(unittest.TestCase):
    def setUp(self) -> None:
        self.loss_tracker = LossTracker(["train_loss", "val_loss"])

    def test_initiation(self):
        self.loss_tracker.reset()
        self.assertEqual(self.loss_tracker.epoch_losses, {"train_loss": [], "val_loss": []})
        self.assertEqual(self.loss_tracker.step_loss_sum, {"train_loss": [], "val_loss": []})

    def test_loss_tracker_step_update(self):
        self.loss_tracker.reset()
        self.loss_tracker.step_update("train_loss", 1)
        self.loss_tracker.step_update("train_loss", 2)
        self.loss_tracker.step_update("val_loss", 3)
        self.loss_tracker.step_update("val_loss", 4)
        self.assertEqual(self.loss_tracker.step_loss_sum["train_loss"], [1, 2])
        self.assertEqual(self.loss_tracker.step_loss_sum["val_loss"], [3, 4])

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

    def test_call(self):
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2  # input, labels (ignore the inputs for this one)
        loss = self.torch_loss_object(model_output, dataloader_data, "train")
        self.assertEqual(loss.item(), 0.0)  # Model output same as labels

    def test_tracked_loss_is_float(self):
        self.torch_loss_object.loss_tracker.reset()
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
        _ = self.torch_loss_object(model_output, dataloader_data, "train")
        loss = self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"]
        self.assertIsInstance(loss[0], float)

    def test_loss_tracker_epoch_ended_correct_length(self):
        test_length = 10
        self.torch_loss_object.loss_tracker.reset()
        model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
        dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
        for i in range(10):
            _ = self.torch_loss_object(model_output, dataloader_data, "train")
            _ = self.torch_loss_object(model_output, dataloader_data, "train")
            self.torch_loss_object.loss_tracker_epoch_ended()
        self.assertEqual(len(self.torch_loss_object.loss_tracker.epoch_losses["train_loss"]), test_length)

    def test_train_validate_modes_write_to_correct_loss(self):
        self.torch_loss_object.loss_tracker.reset()
        for __ in range(2):
            model_output = torch.tensor([1.0, 2.0, 3.0]).cuda()
            dataloader_data = [torch.tensor([1.0, 2.0, 3.0]).cuda()] * 2
            _ = self.torch_loss_object(model_output, dataloader_data, "train")
            self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"], [0.0])
            self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["val_loss"], [])
            _ = self.torch_loss_object(model_output, dataloader_data, "validate")
            self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["train_loss"], [0.0])
            self.assertEqual(self.torch_loss_object.loss_tracker.step_loss_sum["val_loss"], [0.0])
            self.torch_loss_object.loss_tracker_epoch_ended()


class TestBLPathlengthLoss(unittest.TestCase):
    def setUp(self):
        self.dummy_scaler = StandardScaler()  # No scaling
        self.dummy_scaler.scale_ = np.array([1] * 4)
        self.dummy_scaler.mean_ = np.array([0] * 4)
        self.loss_func = BLPathlengthLoss(0, 1, [2, 3], [0, 1], self.dummy_scaler)

    def test_call(self):
        model_output = torch.tensor([1.0, 1.0, 3.0, 4.0]).view(1, -1).cuda()  # mu0, mu1, pathlengthd1, pathlengthd2
        dataloader_data = [
            0,  # dummy
            0,  # dummy
            torch.tensor([1.0, 1.0]).view(1, -1).cuda(),
        ]  # ground truth pulsation ratios
        loss = self.loss_func(model_output, dataloader_data, "train")
        self.assertEqual(loss.item(), 1.0)  # error of 1.0 per detector -> averages to 0.5


class TestSumLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.loss1 = TorchLossWrapper(MSELoss(), "loss1")
        self.loss2 = TorchLossWrapper(MSELoss(), "loss2")
        self.sum_loss = SumLoss([self.loss1, self.loss2], [0.5, 0.5])

    def test_call(self):
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        loss = self.sum_loss(model_output, dataloader_data, "train")
        # MSE -> 1^2 per loss -> 0.5 x 1 + 0.5 x 1 = 1
        self.assertEqual(loss.item(), 1.0)  # Model output same as labels

    def test_tracked_losses(self):
        expected_losses = self.loss1.loss_tracker.tracked_losses + self.loss2.loss_tracker.tracked_losses
        self.assertEqual(self.sum_loss.loss_tracker.tracked_losses, expected_losses)

    def test_losses_are_tracked_properly(self):
        self.sum_loss.loss_tracker.reset()
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        self.sum_loss(model_output, dataloader_data, "train")
        self.sum_loss.loss_tracker_epoch_ended()
        self.assertEqual(self.sum_loss.loss_tracker.step_loss_sum["loss1_train_loss"], [1.0])

    # def test_weights_are_applied_properly(self):
    #     loss1 = TorchLossWrapper(MSELoss(), "loss1")
    #     loss2 = TorchLossWrapper(MSELoss(), "loss2")
    #     sum_loss = SumLoss([self.loss1, self.loss2], [1, 0.5])
    #     model_output = torch.tensor([1.0]).cuda()
    #     dataloader_data = [torch.tensor([2.0]).cuda()] * 2
    #     loss = sum_loss(model_output, dataloader_data, "train")
    #     self.assertEqual(sum_loss.loss_tracker.per_step_losses["loss1_train_loss"], [1.0])
    #     self.assertEqual(sum_loss.loss_tracker.per_step_losses["loss2_train_loss"], [0.5])


class TestTorchLossWithChangingWeight(unittest.TestCase):
    def setUp(self) -> None:
        loss_func = TorchLossWrapper(MSELoss())
        self.loss_nodelay = TorchLossWithChangingWeight(loss_func, 0, 1, 2)
        self.loss_delay = TorchLossWithChangingWeight(loss_func, 1, 2, 2, 3)

    def test_weights_applied_correctly(self):
        self.loss_nodelay.reset()
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        _ = self.loss_nodelay(model_output, dataloader_data, "train")
        self.loss_nodelay.loss_tracker_epoch_ended()
        _ = self.loss_nodelay(model_output, dataloader_data, "train")
        self.loss_nodelay.loss_tracker_epoch_ended()
        # First loss = 1, weight = 0, second loss = 1, weight = 1 -> epoch loss = [0.0, 1.0]
        self.assertEqual(self.loss_nodelay.loss_tracker.epoch_losses["train_loss"], [0.0, 1.0])

    def test_current_epoch_counter(self):
        self.loss_nodelay.reset()
        self.assertEqual(self.loss_nodelay.current_epoch, 0)  # Starts at 1
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        _ = self.loss_nodelay(model_output, dataloader_data, "train")
        self.loss_nodelay.loss_tracker_epoch_ended()
        self.assertEqual(self.loss_nodelay.current_epoch, 1)  # First epoch
        _ = self.loss_nodelay(model_output, dataloader_data, "train")
        self.loss_nodelay.loss_tracker_epoch_ended()
        self.assertEqual(self.loss_nodelay.current_epoch, 1)  # Epoch counter should not increase
        _ = self.loss_nodelay(model_output, dataloader_data, "train")
        self.loss_nodelay.loss_tracker_epoch_ended()
        self.assertEqual(self.loss_nodelay.current_epoch, 1)  # Epoch counter should not increase
    
    def test_delayed_weight_change(self):
        self.loss_delay.reset()
        model_output = torch.tensor([1.0]).cuda()
        dataloader_data = [torch.tensor([2.0]).cuda()] * 2
        for i in range(5):
            _ = self.loss_delay(model_output, dataloader_data, "train")
            self.loss_delay.loss_tracker_epoch_ended()
        # First three epochs: weight = 1, next two epochs: starts with weight = 1, ends with weight = 2
        self.assertEqual(self.loss_delay.loss_tracker.epoch_losses["train_loss"], [1.0, 1.0, 1.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
