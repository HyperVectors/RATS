import unittest
import ratspy as rp
import numpy as np


class TestRATSPy(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.features = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]],
            dtype=np.float64,
        )
        self.labels = ["0", "1", "0", "0"]

    def test_dataset_loading(self):
        # Test if the dataset loads correctly
        dataset = rp.Dataset(self.features, self.labels)
        self.assertEqual(len(dataset.features), len(self.features))
        self.assertEqual(len(dataset.labels), len(self.labels))

    def test_augmentation_pipeline(self):
        # Test if the augmentation pipeline works correctly

        dataset = rp.Dataset(self.features, self.labels)
        pipeline = (
            rp.AugmentationPipeline()
            + rp.Repeat(times=2)
            + rp.Rotation(anchor=0.5)
            + rp.Scaling(min=0.5, max=1.5)
            + rp.Convolve(rp.ConvolveWindow.Gaussian, size=31)
            + rp.Drift(max_drift=1.0, n_drift_points=5)
            + rp.Jittering(standard_deviation=0.1)
            + rp.Drop(percentage=0.1, default=0.0)
            + rp.Crop(size=64)
            + rp.AmplitudePhasePerturbation(
                magnitude_std=-10.0, phase_std=1.7, is_time_domain=True
            )
            + rp.Quantize(levels=50)
            + rp.Reverse()
            + rp.Permutate(window_size=50, segment_size=5)
            + rp.AddNoise(noise_type=rp.NoiseType.Gaussian, mean=0.0, std_dev=0.1)
            + rp.RandomTimeWarpAugmenter(window_size=0, speed_ratio_range=(0.5, 1.5))
        )

        pipeline.augment_batch(dataset, parallel=True)

        # Check if the augmented dataset has the same length
        self.assertEqual(len(dataset.features), 2 * len(self.features))
        self.assertEqual(len(dataset.labels), 2 * len(self.labels))

    def test_fft(self):
        dataset = rp.Dataset(self.features, self.labels)

        frequencies = rp.Transforms.fft(dataset, parallel=True)

        self.assertEqual(len(frequencies.features), len(dataset.features))

        inverse_dataset = rp.Transforms.ifft(frequencies, parallel=True)

        self.assertEqual(len(inverse_dataset.features), len(dataset.features))

        max_diff, is_valid = rp.Transforms.compare_within_tolerance(
            dataset, inverse_dataset, tolerance=1e-6
        )
        self.assertTrue(is_valid, f"Max difference {max_diff} exceeds tolerance")

    def test_dct(self):
        dataset = rp.Dataset(self.features, self.labels)

        dct_features = rp.Transforms.dct(dataset, parallel=True)

        self.assertEqual(len(dct_features.features), len(dataset.features))

        inverse_dct = rp.Transforms.idct(dct_features, parallel=True)

        self.assertEqual(len(inverse_dct.features), len(dataset.features))

        max_diff, is_valid = rp.Transforms.compare_within_tolerance(
            dataset, inverse_dct, tolerance=1e-6
        )
        self.assertTrue(is_valid, f"Max difference {max_diff} exceeds tolerance")

    def test_dtw(self):
        dataset = rp.Dataset(self.features, self.labels)

        dtw_distance, path = rp.QualityBenchmarking.compute_dtw(
            dataset.features[0], dataset.features[1]
        )

        self.assertIsInstance(dtw_distance, float)
        self.assertIsInstance(path, list)
        self.assertGreaterEqual(dtw_distance, 0.0)
        self.assertEqual(len(path), len(dataset.features[0] - 1))


if __name__ == "__main__":
    unittest.main(exit=False)
