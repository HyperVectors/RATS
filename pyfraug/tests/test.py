import numpy as np
import pyfraug as pf

x = np.array([1., 2., 3.])

dataset = pf.Dataset(np.array([[1., 2., 3.], [4., 5., 6.]]), ["1", "2"])

aug = pf.AddNoise(pf.NoiseType.Uniform, bounds=(2.0,3.0))

aug.augment_dataset(dataset)

print(dataset.features)
print(dataset.labels)