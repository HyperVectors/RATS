# pyFraug
pyFraug provides Python bindings for `fraug`, a rust crate for **f**ast **r**ust-based time series **aug**mentation on labeled univariate time series data.

See the original [`fraug` documentation](https://effairust2025-031aba.pages.rwth-aachen.de/) for a more detailed overview of the library.

## Installation
An installation using the Python packaging index is currently not possible, meaning that the package needs to be built manually.

## Build instructions
1. Clone the repository
2. cd to the `pyfraug` directory
3. Install all required Python packages from `requirements.txt` in a virtual environment of your choice: `pip install -r requirements.txt`
4. Now, to build the project, use maturin: `maturin develop -r`
    - Maturin automatically installs the resulting binary in your Python environment

## Usage
Example to reconstruct the surface from an input file, apply some post-processing methods and write the data back to a file:
```python
import pyfraug as pf
import numpy as np

# Dataset from https://timeseriesclassification.com/dataset.php
data = pd.read_csv("./data/Car/Car.csv").to_numpy()

x = data[:,:-1].astype(np.float64)
y = list(map(lambda a: str(a), data[:,-1]))

dataset = pf.Dataset(x, y)

addnoise = pf.AddNoise(pf.NoiseType.Slope, bounds=(0.01, 0.05))
# Only execute the AddNoise augmenter for half of the series in the dataset
addnoise.probability = 0.5

pipeline = (pf.AugmentationPipeline()
            + pf.Repeat(10)
            + pf.Crop(100)
            + addnoise
            + pf.Jittering(0.1))

pipeline.augment_batch(dataset, parallel=True)

# Access augmented data using dataset.features and dataset.labels
```

## Development notes
### Stub file generation
To automatically generate a stub file (`pyfraug.pyi`) for the package, run `cargo run --bin stub_gen`.

This will expose Rust documentation comments to the Python side as docstrings.

### Sphinx Documentation
To generate the Sphinx documentation, make sure that the package is installed through, e.g., maturin, and then run make html in the `docs` directory.
The resulting HTML files will be in `docs/_build/html`.

You also need to install the documentation dependencies from `docs/requirements.txt`.

### Benchmarking
In `benchmarking/` are scripts that automatically benchmark this library against the Python-native library [`tsaug`](https://tsaug.readthedocs.io/en/stable/). 
Which augmenters should be benchmarked can be configured using the `benchmarking/augmenter_configs.yaml` file.
