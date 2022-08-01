GNINA_ Models
=============

Thanks to `Andrew McNutt`_, who converted the original Caffe_ models to PyTorch_, all GNINA_ models are available in gninatorch_.

You can find more information about the Caffe_ implementation at `gnina/models`_.

Loading GNINA_ Models
---------------------

The pre-trained models can be easily loaded as follows:

.. code-block:: python

    from gninatorch import gnina

    model, ensemble: bool = setup_gnina_model(model_name)

where :code:`model_name` corresponds accepts the same values as the :code:`--cnn` argument in GNINA_.

:code:`ensemble` is a boolean flag that indicates whether the model is an ensemble of models or not.

.. warning::
    In contrast to GNINA_, which returns :code:`CNNscore`, the PyTorch models return :code:`log_CNNscore`.

The following models are provided:

* :code:`default2017` :cite:`ragoza2017protein`
* :code:`redock_default2018` or :code:`redock_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`general_default2018` or :code:`general_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`crossdock_default2018` or :code:`crossdock_default2018_[1-4]` :cite:`francoeur2020three`
* :code:`dense` or :code:`dense_[1-4]` :cite:`francoeur2020three`

The following ensembles of models are also provided:

* :code:`default` (GNINA_ default model) :cite:`mcnutt2021gnina` :cite:`francoeur2020three`
* :code:`redock_default2018_ensemble` :cite:`francoeur2020three`
* :code:`general_default2018_ensemble` :cite:`francoeur2020three`
* :code:`crossdock_default2018_ensemble` :cite:`francoeur2020three`
* :code:`dense_ensemble` :cite:`francoeur2020three`

:code:`default` is the default model used by GNINA_. See :cite:`mcnutt2021gnina` for more information.

.. note::
    If you are using the pre-trained models, please cite accordingly.

Building your own ensemble
--------------------------

You can build your own ensemble of models as follows:

.. code-block:: python

    from gninatorch import gnina

    model = gnina.setup_gnina_models([model_name1, model_name2, ...])

The :code:`default` model used by GNINA_ corresponds to the following ensemble:

.. code-block:: python

    from gninatorch import gnina

    names = [
            "dense",
            "general_default2018_3",
            "dense_3",
            "crossdock_default2018",
            "redock_default2018_2",
        ]

    model = gnina.load_gnina_models(names)

The :code:`default` optimises accuracy and inference speed. See :cite:`mcnutt2021gnina` for more information.

Inference with GNINA_ Models
----------------------------

Inference with the pre-trained GNINA_ models is provided by :code:`gninatorch.gnina`:

.. code-block:: bash

    python -m gninatorch.gnina -h

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
.. _`Andrew McNutt`: https://github.com/drewnutt/
.. _Caffe: http://caffe.berkeleyvision.org/
.. _`gnina/models`: https://github.com/gnina/models
