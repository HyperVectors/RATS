from random import random
from .pyfraug import *

class AugmentationPipeline:

    def __init__(self):
        self.augmenters = []

    def __add__(self, other):
        self.augmenters.append(other)

        return self

    def augment_batch(self, dataset: Dataset, *, parallel):
        for augmenter in self.augmenters:
            augmenter.augment_batch(dataset, parallel=parallel)

    def augment_one(self, x):
        res = x
        for augmenter in self.augmenters:
            res = augmenter.augment_one(res)

        return res