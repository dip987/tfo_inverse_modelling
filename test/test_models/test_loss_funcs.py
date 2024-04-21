import unittest
from inverse_modelling_tfo.models.loss_funcs import LossTracker, BLPathlengthLoss, TorchLossWrapper


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
        self.loss_tracker.epoch_update(2)
        self.assertEqual(self.loss_tracker.epoch_losses["train_loss"], 1.5)
        self.assertEqual(self.loss_tracker.epoch_losses["val_loss"], 3.5)


if __name__ == "__main__":
    unittest.main()
