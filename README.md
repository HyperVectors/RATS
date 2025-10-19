# High Performance Time Series Augmentation Libraries (RATS & RATSpy)

This repository contains the code for **RATS** - a rust-based rapid time-series data augmentation library and its python wrapper - **RATSpy**. Data augmentation has been an important technique used for preparing time-series data to improve model generalization and inference capabilities, especially on unseen datasets.

## Fast Rust-based Augmentation

We introduce **RA**pid **T**ime **S**eries **Aug**mentation- **RATS** , a high-performance time-series data augmentation crate developed in Rust. RATS is available as a Rust-crate and leverages rust's core features as well as parallelism to speed-up the augmentation process. With this library, we solve the most common problem with respect to time-series data-augmentation: Performance bottlenecks! To enable cross-language integration, we have also developed a python wrapper for RATS, namely, **RATSpy**. 

## Benchmarking 

The popular python-based time-sries library [tsaug](https://tsaug.readthedocs.io/en/stable/index.html) is used as a reference for comparing and benchmarking performance characteristics. On benchmarking against numerous datasets from the [UCR](https://www.timeseriesclassification.com/dataset.php) archives, we show that our libraries are faster in performing the corresponding augmentations. At the same time, we also implement some augmentations which are not available in tsaug as added features. 

## Usage

Our Rust crate can be found in the [rats](rats/) directory and the Python bindings in [ratspy](ratspy/). All additional information regarding **installation**, **usage** and a **detailed documentation** of both these libraries can be found in the corresponding READMEs for [rats](rats/README.md) and [ratspy](ratspy/README.md).

## Acknowledgement

This work is done in co-operation with the [Chair of AI Methodology](https://www.aim.rwth-aachen.de/)) at the RWTH Aachen University.

## Contributors

This work is authored by [Felix Kern](https://git.rwth-aachen.de/felixkern04), [Aaryaman Basu Roy](https://git.rwth-aachen.de/basuroyaryamaan), [Tejas Pradhan](https://git.rwth-aachen.de/pradhan.tejas135) and [Wadie Skaf](https://github.com/wadieskaf).
