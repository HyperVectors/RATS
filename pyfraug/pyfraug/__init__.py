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
        self.per_sample = False

    def __add__(self, other):
        self.augmenters.append(other)
        return self

    def augment_batch(self, dataset: Dataset, *, parallel, per_sample=False):
        r"""Augment a whole batch

        Parallelized when `parallell` is set
        """
        if per_sample:
            for augmenter in self.augmenters:
                # Compatibility check: reject if any augmenter has per-sample chaining disabled in pipeline
                if not augmenter.supports_per_sample():
                    raise RuntimeError(
                        f"Augmenter '{getattr(augmenter, 'get_name', lambda: type(augmenter).__name__)()}' "
                        "is not compatible with per-sample pipelining!"
                    )
        for augmenter in self.augmenters:
            augmenter.augment_batch(dataset, parallel=parallel, per_sample=per_sample)

    def augment_one(self, x):
        r"""Augment one time series

        When called, the augmenter will always augment the series no matter what the probability for this augmenter is
        """
        res = x
        for augmenter in self.augmenters:
            res = augmenter.augment_one(res)

        return res