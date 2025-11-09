# High Performance Time Series Augmentation Libraries (RATS & RATSpy)

[![ReadTheDocs](https://img.shields.io/badge/Readthedocs-%23000000.svg?style=for-the-badge&logo=readthedocs&logoColor=white)](https://effairust2025-031aba.pages.rwth-aachen.de/)


This repository contains the code for **RATS** - a rust-based rapid time-series data augmentation library and its python wrapper - **RATSpy**. 

**R**apid **A**ugmentations for **T**ime **S**eries - **RATS** is a high-performance time-series data augmentation crate developed in Rust. RATS is available as a Rust crate and leverages Rust's core features as well as parallelism to speed-up the augmentation process. This library addresses the most common problem with respect to time-series data-augmentation: Performance bottlenecks. To enable cross-language integration, a python wrapper for RATS has been developed, namely, **RATSpy**. 

## Benchmarking 

The popular python-based time-series library [tsaug](https://tsaug.readthedocs.io/en/stable/index.html) is used as a reference for comparing and benchmarking performance characteristics. On benchmarking against numerous datasets from the [UCR](https://www.timeseriesclassification.com/dataset.php) archives, RATS and RATSpy are shown to be faster in performing the corresponding augmentations. At the same time, some augmentations are implemented which are not available in tsaug as added features. 

## Project Structure

The RATS crate provides many different augmenters. All augmenters are implemented in their own file in the `augmenters` module. They all implement the `Augmenter` trait which is implemented in `src/augmenters/base.rs`, allowing a common interface and the incorporation into an `AugmentationPipeline` which executes many arbitrary augmenters at once.

All augmenters work on the `Dataset` struct which holds a dataset or a batch of labeled univariate time series data.

## Usage

The Rust crate can be found in the [rats](rats/) directory and the Python bindings in [ratspy](ratspy/). All additional information regarding **installation**, **usage** and a **detailed documentation** of both these libraries can be found in the corresponding READMEs for [rats](rats/README.md) and [ratspy](ratspy/README.md).

## Contributors

[Felix Kern](https://git.rwth-aachen.de/felixkern04), [Aaryaman Basu Roy](https://git.rwth-aachen.de/basuroyaryamaan), [Tejas Pradhan](https://git.rwth-aachen.de/pradhan.tejas135) and [Wadie Skaf](https://github.com/wadieskaf).
