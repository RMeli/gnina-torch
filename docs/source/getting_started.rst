Getting Started
===============

This page details how to get started with _gninatorch.

Introduction
------------

gninatorch_ is a PyTorch_ implementation of GNINA_ scoring function, a CNN-based scoring function for molecular docking.

.. note::
    gninatorch_ depends on libmolgrid_, and therefore it is only available on Linux and requires a NVIDIA_ GPU.

Installation
------------

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/RMeli/gnina-torch.git
    cd gnina-torch

Create conda_ or mamba_ environment with all the dependencies:

.. code-block:: bash

    conda env create -f devtools/conda-envs/gninatorch.yaml
    conda activate gninatorch

Install gninatorch_ from source:

.. code-block:: bash

    python -m pip install .


Testing the Installation
~~~~~~~~~~~~~~~~~~~~~~~~

Run tests with pytest_ and report code coverage:

.. code-block:: bash

    pytest --cov=gninatorch

Loading GNINA Models
--------------------

Thanks to `Andrew McNutt`_, who converted the original GNINA_ Caffe_ models to PyTorch_, all GNINA_ models are available in gninatorch_.
The pre-trained models can be easily loaded as follows:

.. code-block:: python

    from gninatorch import gnina

    model, ensemble: bool = setup_gnina_model(model_name)

where :code:`model_name` corresponds accepts the same values as the :code:`--cnn` argument in GNINA_.

For single models we have the following possibilities:

* :code:`default2017` :cite:`ragoza2017protein`
* :code:`redock_default2018` or :code:`redock_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`general_default2018` or :code:`general_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`crossdock_default2018` or :code:`crossdock_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`dense` or :code:`dense_[1-4]` :cite:`francoeur2020three`

For ensembles of 5 models we have the following possibilities:

* :code:`default` (GNINA_ default model) :cite:`mcnutt2021gnina` :cite:`francoeur2020three`
* :code:`redock_default2018_ensemble` :cite:`francoeur2020three`
* :code:`general_default2018_ensemble` :cite:`francoeur2020three`
* :code:`crossdock_default2018_ensemble` :cite:`francoeur2020three`
* :code:`dense_ensemble` :cite:`francoeur2020three`

Inference with GNINA Models
---------------------------

Inference with the pre-trained GNINA_ models is provided by :code:`gninatorch.gnina`:

.. code-block:: bash

    python -m gninatorch.gnina -h


.. _GNINA: https://github.com/gnina/gnina
.. _conda: https://docs.conda.io/en/latest/
.. _mamba: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
.. _gninatorch: https://gnina-torch.readthedocs.io/en/latest/index.html
.. _libmolgrid: https://gnina.github.io/libmolgrid/
.. _NVIDIA: https://www.nvidia.com/
.. _PyTorch: https://pytorch.org/
.. _pytest: https://docs.pytest.org/en/7.1.x/contents.html
.. _`Andrew McNutt`: https://github.com/drewnutt/
.. _Caffe: http://caffe.berkeleyvision.org/

.. bibliography:: references.bib
