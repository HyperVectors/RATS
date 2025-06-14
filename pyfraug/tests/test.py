import numpy as np
import pyfraug as pf

x = np.array([1., 2., 3.])

dataset = pf.Dataset(np.array([[1., 2., 3.], [4., 5., 6.]]), ["1", "2"])

aug = pf.AddNoise(pf.NoiseType.Uniform, bounds=(2.0,3.0))

caug = pf.ConditionalAugmenter(aug, 0.5)

pipeline = pf.AugmentationPipeline() + pf.Repeat(5) + caug

pipeline.augment_dataset(dataset)

print(dataset.features)
print(dataset.labels)