import unittest

try:
    import numpy as np
    from calibrated_protocol import (
        build_matched_control_table,
        compute_ade_fde,
        cv_dead_reckoning,
        evaluate_calibration_criteria,
    )
except ModuleNotFoundError as exc:
    raise unittest.SkipTest(f"Dependency missing for calibrated protocol tests: {exc}")


class CalibratedProtocolTests(unittest.TestCase):
    def test_compute_ade_fde_zero_error(self):
        y = np.zeros((3, 50, 2), dtype=np.float32)
        ade, fde = compute_ade_fde(y, y)
        self.assertEqual(ade, 0.0)
        self.assertEqual(fde, 0.0)

    def test_cv_dead_reckoning_constant_velocity(self):
        x = np.zeros((1, 30, 7), dtype=np.float32)
        x[0, -2, 2] = 10.0
        x[0, -1, 2] = 11.0  # 10 m/s if dt=0.1
        x[0, -2, 3] = 4.0
        x[0, -1, 3] = 4.5  # 5 m/s if dt=0.1

        pred = cv_dead_reckoning(x)
        self.assertEqual(pred.shape, (1, 50, 2))
        np.testing.assert_allclose(pred[0, 0], np.array([12.0, 5.0]), atol=1e-6)
        np.testing.assert_allclose(pred[0, 49], np.array([61.0, 29.5]), atol=1e-6)

    def test_sigma_ratio_inconclusive_when_lane_change_n_below_threshold(self):
        n = 150
        clean_y = np.zeros((n, 50, 2), dtype=np.float32)
        clean_x = np.zeros((n, 30, 7), dtype=np.float32)
        preds = np.zeros_like(clean_y)

        criteria, verdict, aux = evaluate_calibration_criteria(
            clean_y=clean_y,
            lstm_preds=preds,
            clean_x=clean_x,
            lane_width_m=3.7,
        )

        sigma = [c for c in criteria if c.name == "lateral_endpoint_sigma_ratio"][0]
        self.assertEqual(sigma.status, "inconclusive")
        self.assertEqual(aux["lane_change_n"], 0)
        self.assertEqual(verdict, "calibration_pass")

    def test_matched_control_definition_has_exactly_one_difference_flag(self):
        x_corr = np.random.randn(4, 30, 7).astype(np.float32)
        x_ch1 = np.random.randn(4, 30, 7).astype(np.float32)
        npz = {"X_corrupted": x_corr, "X_ch1_imputed": x_ch1}

        conditions, pairs = build_matched_control_table(npz)
        self.assertEqual(len(conditions), 2)
        self.assertEqual(len(pairs), 1)
        self.assertTrue(bool(pairs.iloc[0]["exactly_one_difference"]))


if __name__ == "__main__":
    unittest.main()
