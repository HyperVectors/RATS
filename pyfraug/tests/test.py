import time

import pyfraug as pf
import pandas as pd
import numpy as np

data = pd.read_csv("../../data/InsectSound/InsectSound.csv").to_numpy()

x = data[:,:-1].astype(np.float64)
y = list(map(lambda a: str(a), data[:,-1]))

dataset = pf.Dataset(x, y)

# pipeline = (pf.AugmentationPipeline()
#             + pf.Repeat(10)
#             + pf.Crop(100)
#             + pf.ConditionalAugmenter(
#                 pf.AddNoise(pf.NoiseType.Slope, bounds=(0.01, 0.02)),
#                 0.5
#             )
#             + pf.Jittering(0.1))
aug = pf.AugmentationPipeline() + pf.AddNoise(pf.NoiseType.Gaussian, mean=0, std_dev=0.1)

start = time.time()
aug.augment_dataset(dataset)
end = time.time()
print(f"Pyfraug took {end - start:.4f} seconds")


from tsaug import AddNoise

my_augmenter = AddNoise(scale=0.1)

start = time.time()
my_augmenter.augment(x)
end = time.time()
print(f"Tsaug took {end - start:.4f} seconds")