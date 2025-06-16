import pyfraug as pf
import pandas as pd

data = pd.read_csv("../data/Car/Car.csv").to_numpy()

x = data[:,:-1]
y = list(map(lambda a: str(a), data[:,-1]))

dataset = pf.Dataset(x, y)

pipeline = (pf.AugmentationPipeline()
            + pf.Repeat(10)
            + pf.Crop(100)
            + pf.ConditionalAugmenter(
                pf.AddNoise(pf.NoiseType.Slope, bounds=(0.01, 0.02)),
                0.5
            )
            + pf.Jittering(0.1))

pipeline.augment_dataset(dataset)
