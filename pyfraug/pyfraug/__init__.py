from random import random
from .pyfraug import *

class ConditionalAugmenter:

    def __init__(self, augmenter, probability):
        self.augmenter = augmenter
        self.probability = probability

    def augment_dataset(self, dataset: Dataset, *, parallel):
        features = dataset.features
        for row in features:
            self.augment_one(row)
        dataset.features = features

    def augment_one(self, x):
        if random() < self.probability:
            self.augmenter.augment_one(x)

class AugmentationPipeline:

    def __init__(self):
        self.augmenters = []

    def __add__(self, other):
        # if not issubclass(type(other), Augmenter):
        #     print("Not an augmenter!")
        #     return self

        self.augmenters.append(other)

        return self

    def augment_dataset(self, dataset: Dataset, *, parallel):
        for augmenter in self.augmenters:
            augmenter.augment_dataset(dataset, parallel=parallel)

    def augment_one(self, x):
        for augmenter in self.augmenters:
            augmenter.augment_one(x)