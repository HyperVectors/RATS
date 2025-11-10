# RATSpy
RATSpy provides Python bindings for `RATS`, a rust crate for **R**apid **A**ugmentations for **T**ime **S**eries on univariate time series data.

See the original [`RATS` documentation](https://effairust2025-031aba.pages.rwth-aachen.de/) for a more detailed overview of the library.

## Installation
An installation using the Python packaging index is currently not possible, meaning that the package needs to be built manually.

## Build instructions
1. Clone the repository
2. cd to the `ratspy` directory
3. Install all required Python packages from `requirements.txt` in a virtual environment of your choice: `pip install -r requirements.txt`
4. Now, to build the project, use maturin: `maturin develop -r`
    - Maturin automatically installs the resulting binary in your Python environment

## Usage
Example to reconstruct the surface from an input file, apply some post-processing methods and write the data back to a file:
```python
import ratspy as rp
import numpy as np

# Dataset from https://timeseriesclassification.com/dataset.php
data = pd.read_csv("./data/Car/Car.csv").to_numpy()

x = data[:,:-1].astype(np.float64)
y = list(map(lambda a: str(a), data[:,-1]))

dataset = rp.Dataset(x, y)

addnoise = rp.AddNoise(rp.NoiseType.Slope, bounds=(0.01, 0.05))
# Only execute the AddNoise augmenter for half of the series in the dataset
addnoise.probability = 0.5

pipeline = (rp.AugmentationPipeline()
            + rp.Repeat(10)
            + rp.Crop(100)
            + addnoise
            + rp.Jittering(0.1))

pipeline.augment_batch(dataset, parallel=True)

# Access augmented data using dataset.features and dataset.labels
```

## Development notes
### Stub file generation
To automatically generate a stub file (`ratspy.pyi`) for the package, run `cargo run --bin stub_gen`.

This will expose Rust documentation comments to the Python side as docstrings.

### Sphinx Documentation
To generate the Sphinx documentation, make sure that the package is installed through, e.g., maturin, and then run make html in the `docs` directory.
The resulting HTML files will be in `docs/_build/html`.

You also need to install the documentation dependencies from `docs/requirements.txt`.

### Tests
To test the installation of ratspy in your `venv` and run some initial tests, we provide unit tests for each of ratspy's classes as well as the entire pipeline in the `/tests` directory. 

Run the following commands to run tests: 

```shell
cd tests
python -m unittest test_RATSpy
```

### Benchmarking
In `benchmarking/` are scripts that automatically benchmark this library against the Python-native library [`tsaug`](https://tsaug.readthedocs.io/en/stable/). 
Which augmenters should be benchmarked can be configured using the `benchmarking/augmenter_configs.yaml` file.
