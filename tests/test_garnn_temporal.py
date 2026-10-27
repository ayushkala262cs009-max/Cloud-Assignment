import unittest

try:
    import torch
    from garnn_temporal import GarnnTemporalBranch, masked_reconstruction_mae
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Dependency missing for GARNN tests: {exc}")


class GarnnTemporalTests(unittest.TestCase):
    def test_garnn_temporal_output_shape_matches_input(self):
        model = GarnnTemporalBranch(feature_dim=7, hidden_dim=16)
        values = torch.randn(5, 30, 7)
        mask = torch.ones(5, 30, 7)

        out = model(values, mask)
        self.assertEqual(out.shape, values.shape)

    def test_masked_mae_uses_only_missing_cells(self):
        pred = torch.tensor([[[2.0], [4.0]]])
        true = torch.tensor([[[1.0], [1.0]]])
        mask = torch.tensor([[[1.0], [0.0]]])

        mae = masked_reconstruction_mae(pred, true, mask)
        self.assertTrue(torch.isclose(mae, torch.tensor(3.0)))

    def test_masked_mae_returns_zero_when_no_missing_cells(self):
        pred = torch.ones(2, 3, 4)
        true = torch.zeros(2, 3, 4)
        mask = torch.ones(2, 3, 4)
        mae = masked_reconstruction_mae(pred, true, mask)
        self.assertEqual(mae.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
