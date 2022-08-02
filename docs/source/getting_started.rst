Getting Started
===============

gninatorch_ is a PyTorch_ implementation of GNINA_ scoring function, a CNN-based scoring
function for molecular docking.

.. note::
    gninatorch_ depends on libmolgrid_, and therefore it is only available on Linux and
    requires a NVIDIA_ GPU.

If you use gninatorch_, please consider citing the following papers:
:cite:`ragoza2017protein`, :cite:`sunseri2020libmolgrid`, :cite:`francoeur2020three`,
and :cite:`mcnutt2021gnina`.

Help
----

If you find an issue with gninatorch_, please open a `GitHub issue`_. If you have a
question about gninatorch_, please use `GitHub Discussions`_.

Installation
------------

Installation from Source
~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository from GitHub_:

.. code-block:: bash

    git clone https://github.com/RMeli/gnina-torch.git
    cd gnina-torch

Create a conda_ or mamba_ environment with all the dependencies:

.. code-block:: bash

    conda env create -f devtools/conda-envs/gninatorch.yaml
    conda activate gninatorch

Install gninatorch_ from source:

.. code-block:: bash

    python -m pip install .


Installation from PyPI
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m pip install gninatorch

.. warning::

    Packages on PyPI are still WIP and should be considered experimental.

Testing
-------

Run tests with pytest_ and report code coverage:

.. code-block:: bash

    pytest --cov=gninatorch

.. raw:: html

   <hr>

.. bibliography::
   :cited:

.. _GNINA: https://github.com/gnina/gnina
.. _conda: https://docs.conda.io/en/latest/
.. _mamba: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
.. _gninatorch: https://gnina-torch.readthedocs.io/en/latest/index.html
.. _libmolgrid: https://gnina.github.io/libmolgrid/
.. _NVIDIA: https://www.nvidia.com/
.. _PyTorch: https://pytorch.org/
.. _pytest: https://docs.pytest.org/en/7.1.x/contents.html
.. _GitHub: https://github.com/
.. _`GitHub issue`: https://github.com/RMeli/gnina-torch/issues
.. _`GitHub Discussions`: https://github.com/RMeli/gnina-torch/discussions