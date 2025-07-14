from random import random
from .pyfraug import *

class AugmentationPipeline:
    r"""A pipeline of augmenters

    Executes many augmenters at once. Append augmenters to the pipeline by adding them to the pipeline:

    ```
    import pyfraug as pf

    pipeline = pf.AugmentationPipeline() + pf.Repeat(5) + pf.Crop(20)
    ```
    """

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