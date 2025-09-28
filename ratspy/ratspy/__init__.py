from random import random
try:
    from .ratspy import *
except ModuleNotFoundError:
    from ._ratspy import *

class AugmentationPipeline:
    r"""A pipeline of augmenters

    Executes many augmenters at once. Append augmenters to the pipeline by adding them to the pipeline:

    ```
    import ratspy as rp

    pipeline = rp.AugmentationPipeline() + rp.Repeat(5) + rp.Crop(20)
    ```
    """

    def __init__(self):
        self.augmenters = []

    def __add__(self, other):
        self.augmenters.append(other)
        return self

    def augment_batch(self, dataset: Dataset, *, parallel):
        r"""Augment a whole batch

        Parallelized when `parallell` is set
        """
        for augmenter in self.augmenters:
            augmenter.augment_batch(dataset, parallel=parallel)

    def augment_one(self, x):
        r"""Augment one time series

        When called, the augmenter will always augment the series no matter what the probability for this augmenter is
        """
        res = x
        for augmenter in self.augmenters:
            res = augmenter.augment_one(res)

        return res