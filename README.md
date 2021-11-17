# gnina-torch

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RMeli/gnina-torch/workflows/CI/badge.svg)](https://github.com/RMeli/gnina-torch/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/RMeli/gnina-torch/branch/main/graph/badge.svg?token=KjVkShwQ1z)](https://codecov.io/gh/RMeli/gnina-torch)

PyTorch implementation of GNINA scoring function.

## Installation

The `gnina` Python package has several dependencies, including:
* [PyTorch](https://pytorch.org/)
* [PyTorch-Ignite](https://pytorch.org/ignite/)
* [libmolgrid](https://gnina.github.io/libmolgrid/)

A full developement environment can be installed using the [conda](https://docs.conda.io/en/latest/) package manager and the provided [conda](https://docs.conda.io/en/latest/) environment file (`devtools/conda-envs/gninatorch.yaml`):
```bash
conda create -f devtools/conda-envs/gninatorch.yaml
conda activate gninatorch
```

Once the [conda](https://docs.conda.io/en/latest/) environment is created and activated, the `gnina` package can be installed using [pip](https://pip.pypa.io/en/stable/) as follows:
```bash
python -m pip install .
```

### Tests

In order to check the installation, unit tests are provided and can be run with [pytest](https://docs.pytest.org/):
```bash
pytest --cov=gnina
```

## Usage

Training and inference modules try to follow the original [Caffe](https://caffe.berkeleyvision.org/) implementation of [gnina/scripts](https://github.com/gnina/scripts), however not all features are implemented.

The folder `examples` includes some complete examples for training and inference.

### Training

Training is implemented in the `training` module:
```bash
python -m gnina.training --help
```

### Inference

Inference is implemented in the `inference` module:
```bash
python -m gnina.inference --help
```

## References

> Protein–Ligand Scoring with Convolutional Neural Networks,
> M. Ragoza, J. Hochuli, E. Idrobo, J. Sunseri, and D. R. Koes, *J. Chem. Inf. Model.* 2017, 57 (4), 942-957.
> DOI: [10.1021/acs.jcim.6b00740](https://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740)

> libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications
> J. Sunseri and D. R. Koes, *J. Chem. Inf. Model.* 2020, 60 (3), 1079–1084.
> DOI: [10.1021/acs.jcim.9b01145](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01145)

---

Copyright (c) 2021, Rocco Meli

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
