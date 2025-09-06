# RATS

[![ReadTheDocs](https://img.shields.io/badge/Readthedocs-%23000000.svg?style=for-the-badge&logo=readthedocs&logoColor=white)](https://effairust2025-031aba.pages.rwth-aachen.de/)

RATS is a **RA**pid **T**ime **S**eries rust based library for augmentations. It is available as a rust crate under `RATS` and as a python package under `RATSpy`.

## Project structure
The crate provides many different augmenters. All of them are implemented in their own file in the `augmenters` module. They all implement the `Augmenter` trait which is implemented in `src/augmenters/base.rs` which allows a common interface and the incorporation into a `AugmentationPipeline` which executes many arbitrary augmenters at once.

In the `transforms` module, functions for frequency domain transformations are provided. These and all augmenters work on the `Dataset` struct which holds a dataset or a batch of labeled univariate time series data. 

## Development notes
### Build instructions
Assuming you have both Rust and cargo installed, building the crate is as simple as `cargo build`. For a more performant library, build it with the release flag set: `cargo build --release`.

### Unit tests
To verify your installation, you can run the unit tests: `cargo test`

### Documentation
To build a local documentation of this crate, run `cargo doc`. However, we also provide a detailed documentation along with usage examples of this library [here](https://effairust2025-031aba.pages.rwth-aachen.de/).