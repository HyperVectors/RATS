from random import random
from .pyfraug import *

class ConditionalAugmenter:

    def __init__(self, augmenter, probability):
        self.augmenter = augmenter
        self.probability = probability

    def augment_dataset(self, dataset: Dataset, *, parallel):
        features = np.array([])
        for row in dataset.features:
            features.append(self.augment_one(row))
        dataset.features = features

    def augment_one(self, x):
        if random() < self.probability:
            return self.augmenter.augment_one(x)

class AugmentationPipeline:

    def __init__(self):
        self.augmenters = []

    def __add__(self, other):
        self.augmenters.append(other)

        return self

    def augment_dataset(self, dataset: Dataset, *, parallel):
        for augmenter in self.augmenters:
            augmenter.augment_dataset(dataset, parallel=parallel)

    def augment_one(self, x):
        res = x
        for augmenter in self.augmenters:
            res = augmenter.augment_one(res)

        return res