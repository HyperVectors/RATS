# High Performance Time Series Augmentation Libraries (fraug & pyFraug)

This is the repository for **Team 02** for the `Efficient AI with Rust` lab in Summer Semester 2025.

## Task Description
The task is to develop a fast, high-performance time series augmentation library in Rust which implements the most commonly used time series augmentations and provide Python bindings for it. 

## Motivation

Just as in any other type of data, augmentation is an important technique used for preparing time-series data to improve model generalization and inference capabilities, especially on unseen datasets. Most python-based time-series augmentation libraries, while accomplishing this task, suffer in speed which becomes a major bottleneck.

## Fast Rust-based Augmentation

We introduce **F**ast **Rust**-based **Aug**mentation- **fraug** , a high-performance time-series data augmentation crate developed in Rust. Fraug is available as a Rust-crate and leverages rust's core features as well as parallelism to speed-up 
the augmentation process. To enable cross-language integration, we have also developed a python wrapper for fraug, namely, **pyFraug**. 

## Benchmarking 

The popular python-based time-sries library [tsaug](https://tsaug.readthedocs.io/en/stable/index.html) is used as a reference for comparing and benchmarking performance characteristics. On benchmarking against numerous datasets from the [UCR](https://www.timeseriesclassification.com/dataset.php) archives, we show that our libraries are faster in performing the corresponding augmentations. At the same time, we also implement some augmentations which are not available in tsaug as added features. 

## Usage

Our Rust crate can be found in the [fraug](fraug/) directory and the Python bindings in [pyfraug](pyfraug/). All additional information regarding **installation**, **usage** and a **detailed documentation** of both these libraries can be found in the corresponding READMEs for [fraug](fraug/README.md) and [pyFraug](pyfraug/README.md).

## Acknowledgments

We would like to thank our supervisors Wadie Skaf and Dr. Roman Kalkreuth for their guidance and support during the course of this project. We also extend our gratitude to the **RWTH Chair of AI Methodology** for facilitating this lab.

## Contributors

This work is authored by [Felix Kern](https://git.rwth-aachen.de/felixkern04), [Aaryaman Basu Roy](https://git.rwth-aachen.de/basuroyaryamaan) and [Tejas Pradhan](https://git.rwth-aachen.de/pradhan.tejas135)