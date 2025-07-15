import time

import pyfraug as pf
import pandas as pd
import numpy as np

data = pd.read_csv("../../data/Car/Car.csv").to_numpy()

x = data[:,:-1].astype(np.float64)
y = list(map(lambda a: str(a), data[:,-1]))

dataset = pf.Dataset(x, y)

aug = pf.AugmentationPipeline() + pf.AddNoise(pf.NoiseType.Gaussian, mean=0, std_dev=0.1) + pf.AmplitudePhasePerturbation(magnitude_std= -10.0, phase_std= 1.7, is_time_domain= True)
start = time.time()
aug.augment_batch(dataset, parallel=True)
end = time.time()
print(f"Augmentation took {end - start:.4f} seconds")