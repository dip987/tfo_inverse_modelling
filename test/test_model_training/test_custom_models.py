import unittest
import torch
from inverse_modelling_tfo.model_training.custom_models import FeatureResidualNetwork


class TestFeatureResidualNetowrk(unittest.TestCase):
    def setUp(self):
        self.in_features = 10
        self.out_features = 5
        self.batch_size = 1
        self.lookup_table = torch.rand(100, self.in_features + self.out_features)
        self.lookup_key_indices = torch.arange(self.in_features, self.out_features + self.in_features)
        self.feature_indices = torch.arange(0, self.in_features)
        self.model = FeatureResidualNetwork(
            [self.in_features, 4, self.lookup_key_indices.shape[0]],
            [0.5, 0.5],
            [self.in_features, 8, self.out_features],
            [0.5, 0.5],
            self.lookup_table,
            self.lookup_key_indices,
            self.feature_indices,
        )
        
        # Training/Testing with only one row of data messes up BatchNorm's moving-mean calcualtion. Set to eval mode to avoid this
        self.model = self.model.eval()

    def test_network_produces_output(self):
        x = torch.rand(self.batch_size, self.in_features)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.out_features))


if __name__ == "__main__":
    unittest.main()
