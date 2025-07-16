import unittest
import pyfraug as pf
import numpy as np


class TestPyfraug(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.features = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
            dtype=np.float64,
        )
        self.labels = ["0", "1", "0", "0"]

    def test_dataset_loading(self):
        # Test if the dataset loads correctly
        dataset = pf.Dataset(self.features, self.labels)
        self.assertEqual(len(dataset.features), len(self.features))
        self.assertEqual(len(dataset.labels), len(self.labels))

    def test_augmentation_pipeline(self):
        # Test if the augmentation pipeline works correctly

        dataset = pf.Dataset(self.features, self.labels)
        pipeline = (
            pf.AugmentationPipeline()
            + pf.Repeat(times=2)
            + pf.Rotation(anchor=0.5)
            + pf.Scaling(min=0.5, max=1.5)
            + pf.Convolve(pf.ConvolveWindow.Gaussian, size=31)
            + pf.Drift(max_drift=1.0, n_drift_points=5)
            + pf.Jittering(standard_deviation=0.1)
            + pf.Drop(percentage=0.1, default=0.0)
            + pf.Crop(size=64)
            + pf.AmplitudePhasePerturbation(
                magnitude_std=-10.0, phase_std=1.7, is_time_domain=True
            )
            + pf.Quantize(levels=50)
            + pf.Reverse()
            + pf.Permutate(size=50)
            + pf.AddNoise(noise_type=pf.NoiseType.Gaussian, mean=0.0, std_dev=0.1)
            + pf.RandomTimeWarpAugmenter(window_size=0, speed_ratio_range=(0.5, 1.5))
        )

        pipeline.augment_batch(dataset, parallel=True)

        # Check if the augmented dataset has the same length
        self.assertEqual(len(dataset.features), 2 * len(self.features))
        self.assertEqual(len(dataset.labels), 2 * len(self.labels))

    def test_fft(self):
        dataset = pf.Dataset(self.features, self.labels)

        frequencies = pf.Transforms.fft(dataset, parallel=True)

        self.assertEqual(len(frequencies.features), len(dataset.features))

        inverse_dataset = pf.Transforms.ifft(frequencies, parallel=True)

        self.assertEqual(len(inverse_dataset.features), len(dataset.features))

        max_diff, is_valid = pf.Transforms.compare_within_tolerance(
            dataset, inverse_dataset, tolerance=1e-6
        )
        self.assertTrue(is_valid, f"Max difference {max_diff} exceeds tolerance")

    def test_dct(self):
        dataset = pf.Dataset(self.features, self.labels)

        dct_features = pf.Transforms.dct(dataset, parallel=True)

        self.assertEqual(len(dct_features.features), len(dataset.features))

        inverse_dct = pf.Transforms.idct(dct_features, parallel=True)

        self.assertEqual(len(inverse_dct.features), len(dataset.features))

        max_diff, is_valid = pf.Transforms.compare_within_tolerance(
            dataset, inverse_dct, tolerance=1e-6
        )
        self.assertTrue(is_valid, f"Max difference {max_diff} exceeds tolerance")

    def test_dtw(self):
        dataset = pf.Dataset(self.features, self.labels)

        dtw_distance, path = pf.QualityBenchmarking.compute_dtw(
            dataset.features[0], dataset.features[1]
        )

        self.assertIsInstance(dtw_distance, float)
        self.assertIsInstance(path, list)
        self.assertGreaterEqual(dtw_distance, 0.0)
        self.assertEqual(len(path), len(dataset.features[0] - 1))


if __name__ == "__main__":
    unittest.main(module="test_pyfraug", exit=False)
