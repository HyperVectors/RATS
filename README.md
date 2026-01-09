# High Performance Time Series Augmentation Libraries (RATS & RATSpy)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2601.03159-b31b1b.svg)](https://arxiv.org/abs/2601.03159)
[![PyPI](https://img.shields.io/pypi/v/ratspy.svg)](https://pypi.org/project/ratspy/)
[![Crates.io](https://img.shields.io/crates/v/rats-rs.svg)](https://crates.io/crates/rats-rs)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/ratspy?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads)](https://pepy.tech/projects/ratspy)

[![ReadTheDocs](https://img.shields.io/badge/Readthedocs-%23000000.svg?style=for-the-badge&logo=readthedocs&logoColor=white)](https://ratspy.readthedocs.io)


This repository contains the code for **RATS** - a rust-based rapid time-series data augmentation library and its python wrapper - **RATSpy**. 

**R**apid **A**ugmentations for **T**ime **S**eries - **RATS** is a high-performance time-series data augmentation crate developed in Rust. RATS is available as a Rust crate and leverages Rust's core features as well as parallelism to speed-up the augmentation process. This library addresses the most common problem with respect to time-series data-augmentation: Performance bottlenecks. To enable cross-language integration, a python wrapper for RATS has been developed, namely, **RATSpy**.

> 📌 If you find this repository helpful or insightful, please consider **giving it a star ⭐**. If you use it in your research, please **cite our paper** (see [Citation](#citation)).

## Benchmarking 

The popular python-based time-series library [tsaug](https://tsaug.readthedocs.io/en/stable/index.html) is used as a reference for comparing and benchmarking performance characteristics. On benchmarking against numerous datasets from the [UCR](https://www.timeseriesclassification.com/dataset.php) archives, RATS and RATSpy are shown to be faster in performing the corresponding augmentations. At the same time, some augmentations are implemented which are not available in tsaug as added features. 

## Project Structure

The RATS crate provides many different augmenters. All augmenters are implemented in their own file in the `augmenters` module. They all implement the `Augmenter` trait which is implemented in `src/augmenters/base.rs`, allowing a common interface and the incorporation into an `AugmentationPipeline` which executes many arbitrary augmenters at once.

All augmenters work on the `Dataset` struct which holds a dataset or a batch of labeled univariate time series data.

## Usage

The Rust crate can be found in the [rats](rats/) directory and the Python bindings in [ratspy](ratspy/). All additional information regarding **installation**, **usage** and a **detailed documentation** of both these libraries can be found in the corresponding READMEs for [rats](rats/README.md) and [ratspy](ratspy/README.md).

## Citation

If you use this repository, adapt any of its components, or find it helpful for your work, please cite:

```bibtex
@misc{skaf-2026-RATS,
  title={Rapid Augmentations for Time Series ({RATS}): A High-Performance Library for Time Series Augmentation},
  author={Skaf, Wadie and Kern, Felix and Basu Roy, Aryamaan and Pradhan, Tejas and Kalkreuth, Roman and Hoos, Holger},
  year={2026},
  eprint={2601.03159},
  archivePrefix={arXiv},
}
```

If this repository is useful to you, a star ⭐ is also very much appreciated and helps others discover the project.

## License

This work is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
