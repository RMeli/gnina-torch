# gnina-torch

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/RMeli/gnina-torch/workflows/CI/badge.svg)](https://github.com/RMeli/gnina-torch/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/RMeli/gnina-torch/branch/main/graph/badge.svg?token=KjVkShwQ1z)](https://codecov.io/gh/RMeli/gnina-torch)

PyTorch implementation of [GNINA](https://github.com/gnina/gnina) scoring function.

## References

If you are using `gnina-torch`, please consider citing the following references:

> Protein–Ligand Scoring with Convolutional Neural Networks,
> M. Ragoza, J. Hochuli, E. Idrobo, J. Sunseri, and D. R. Koes, *J. Chem. Inf. Model.* 2017, 57 (4), 942-957.
> DOI: [10.1021/acs.jcim.6b00740](https://pubs.acs.org/doi/full/10.1021/acs.jcim.6b00740)

> libmolgrid: Graphics Processing Unit Accelerated Molecular Gridding for Deep Learning Applications
> J. Sunseri and D. R. Koes, *J. Chem. Inf. Model.* 2020, 60 (3), 1079–1084.
> DOI: [10.1021/acs.jcim.9b01145](https://pubs.acs.org/doi/10.1021/acs.jcim.9b01145)

If you are using the pre-trained `default2018` and `dense` models from [GNINA](https://github.com/gnina/gnina), please consider citing the following reference as well:

> Three-Dimensional Convolutional Neural Networks and a Cross-Docked Data Set for Structure-Based Drug Design,
> P. G. Francoeur, T. Masuda, J. Sunseri, A. Jia, R. B> Iovanisci, I. Snyder, and D. R. Koes, *J. Chem. Inf. Model.* 2020, 60 (9), 4200-4215.
> DOI: [10.1021/acs.jcim.0c00411](https://doi.org/10.1021/acs.jcim.0c00411)

## Installation

The `gninatorch` Python package has several dependencies, including:

* [PyTorch](https://pytorch.org/)
* [PyTorch-Ignite](https://pytorch.org/ignite/)
* [libmolgrid](https://gnina.github.io/libmolgrid/)

A full developement environment can be installed using the [conda](https://docs.conda.io/en/latest/) package manager and the provided [conda](https://docs.conda.io/en/latest/) environment file (`devtools/conda-envs/gninatorch.yaml`):

```bash
conda env create -f devtools/conda-envs/gninatorch.yaml
conda activate gninatorch
```

Once the [conda](https://docs.conda.io/en/latest/) environment is created and activated, the `gninatorch` package can be installed using [pip](https://pip.pypa.io/en/stable/) as follows:

```bash
python -m pip install .
```

### Tests

In order to check the installation, unit tests are provided and can be run with [pytest](https://docs.pytest.org/):

```bash
pytest --cov=gninatorch
```

## Usage

Training and inference modules try to follow the original [Caffe](https://caffe.berkeleyvision.org/) implementation of [gnina/scripts](https://github.com/gnina/scripts), however not all features are implemented.

The folder `examples` includes some complete examples for training and inference.

The folder `gninatorch/weights` contains pre-trained models from [GNINA](https://github.com/gnina/gnina), converted from Caffe to PyTorch.

### Pre-trained GNINA models

Pre-trained models (`--cnn` argument in [GNINA](https://github.com/gnina/gnina)) can be easily loaded as follows:

```python
from gninatorch.gnina import load_gnina_model

model = load_gnina_model(MODEL_NAME)
```

Inference with pre-trained [GNINA](https://github.com/gnina/gnina) models is implemented in the `gnina` module:

```bash
python -m gninatorch.gnina --help
```

### Training

Training is implemented in the `training` module:

```bash
python -m gninatorch.training --help
```

### Inference

Inference is implemented in the `inference` module:

```bash
python -m gninatorch.inference --help
```

## Acknowledgments

Project based on the [Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

The pre-trained weights of [GNINA](https://github.com/RMeli/gnina-torch) converted to PyTorch were kindly provided by Andrew McNutt (@drewnutt).

---

Copyright (c) 2021, Rocco Meli
