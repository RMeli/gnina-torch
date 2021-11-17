from typing import Optional

import torch


class GriddedExamplesLoader:
    """
    Load example and compute atomic density on a grid.

    Parameters
    ----------
    example_provider : :class:`molgrid.ExampleProvider`
        :package:`molgrid` example provider
    grid_maker : :class:`molgrid.GridMaker`
        :package:`molgrid` grid maker
    label_pos: int
        Ligand pose annotation label position
    affinity_pos: Optional[int]
        Affinity annotation position
    random_translation : float
        Random translation applied to each example on each cartesian axis
    random_rotation : bool
        Uniform random rotation applied to each example
    device : torch.device
        Device

    Notes
    -----
    The batch size is defined in the :code:`example_provider` as
    :code:`default_batch_size`. The number of batches actually depend on the
    :code:`molgrid.IterationScheme` used, also defined in the
    :code:`example_provider`.

    If :code:`molgrid.IterationScheme.SmallEpoch` is used, examples are seen at most
    once. If :code:`molgrid.IterationScheme.LargeEpoch` is used, examples are seen
    at least once.

    The last batch is not padded with examples of the next epoch, in contrast with
    :code:`molgrid.ExampleProvider` default behaviour.
    """

    def __init__(
        self,
        example_provider,
        grid_maker,
        label_pos: int = 0,
        affinity_pos: Optional[int] = None,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        # Check that example provider is populated
        assert example_provider.size() > 0

        self.example_provider = example_provider
        self.grid_maker = grid_maker
        self.label_pos = label_pos
        self.affinity_pos = affinity_pos
        self.random_translation = random_translation
        self.random_rotation = random_rotation
        self.device = device

        # Total number of examples in file
        # This is not necessarily the same as the number of examples seen in an epoch
        # The number of examples in an epoch depends on the example provider settings
        self.num_examples_tot = self.example_provider.size()

        self.num_labels = self.example_provider.num_labels()
        self.num_types = self.example_provider.num_types()

        self.dims = grid_maker.grid_dimensions(self.num_types)

        example_provider_settings = example_provider.settings()
        self.iteration_scheme = str(example_provider_settings.iteration_scheme)

        if self.iteration_scheme == "SmallEpoch":
            self.num_examples_per_epoch = example_provider.small_epoch_size()
        elif self.iteration_scheme == "LargeEpoch":
            self.num_examples_per_epoch = example_provider.large_epoch_size()
        else:
            raise ValueError(f"Unknown iteration scheme {self.iteration_scheme}")

        self.batch_size = example_provider_settings.default_batch_size

        if example_provider_settings.balanced and self.batch_size == 1:
            raise ValueError("Balanced batches incompatible with batch size 1.")

        self.num_batches = (self.num_examples_per_epoch) // self.batch_size
        self.last_batch_size = self.num_examples_per_epoch % self.batch_size
        if self.last_batch_size != 0:
            self.num_batches += 1

        self.batch_idx = 0
        self.last_epoch = False

    # TODO: Check this is what we want
    # TODO: By default, epoch length is defined by len(data)
    # TODO: If data is a finite data iterator with unknown length epoch_length
    # TODO: will be automatically determined when data iterator is exhausted.
    # def __len__(self):
    #     """
    #     Return length of the epoch (number of examples).

    #     Notes
    #     -----
    #     The number of examples per epoch depends on the :code:`molgrid.IterationScheme`
    #     used. Without balancing nor stratification, the number of examples per epoch is
    #     the same for :code:`molgrid.IterationScheme.SmalleEpoch` and
    #     :code:`molgrid.IterationScheme.LargeEpoch`. For balanced sampling, which sample
    #     the same number of positive and negative examples, the number of examples in a
    #     small epoch (examples seen at most once) is twice the size of the minority class
    #     while for a large epoch (examples seen at least once) it is twice the size of
    #     the majority class.
    #     """
    #     settings = self.example_provider.settings()

    #     if settings.iteration_scheme == molgrid.IterationScheme.SmallEpoch:
    #         return self.example_provider.small_epoch_size()
    #     elif settings.iteration_scheme == molgrid.IterationScheme.LargeEpoch:
    #         return self.example_provider.large_epoch_size()
    #     else:
    #         raise ValueError("Unknown iteration scheme {settings.iteration_scheme}.")

    def __next__(self):
        """
        Get next batch of gridded examples and corresponding labels.
        """
        if self.last_epoch:
            raise StopIteration

        if self.batch_idx < self.num_batches - 1:  # All batches except the last
            batch = self.example_provider.next_batch(self.batch_size)
            self.batch_idx += 1
        else:  # Treat last batch differently
            if self.last_batch_size != 0:  # Last batch has a different size
                # This avoids padding with examples from the next epoch
                batch = self.example_provider.next_batch(self.last_batch_size)
            else:  # Last batch has the same size as previous batches
                batch = self.example_provider.next_batch(self.batch_size)
            # Last epoch, raise StopIteration at the next iteration attempt
            # Reset index
            self.last_epoch = True

        batch_size = len(batch)

        grids = torch.zeros((batch_size, *self.dims), device=self.device)
        labels = torch.zeros((batch_size,), device=self.device)
        if self.affinity_pos is not None:
            affinities = torch.zeros((batch_size,), device=self.device)

        # Compute grids from examples
        self.grid_maker.forward(
            batch,
            grids,
            random_translation=self.random_rotation,
            random_rotation=self.random_rotation,
        )

        batch.extract_label(self.label_pos, labels)
        if self.affinity_pos is not None:
            batch.extract_label(self.affinity_pos, affinities)

        # Convert labels to integers
        # libmolgrid only supports float input
        labels = labels.long()

        if self.affinity_pos is None:
            # Return grids and labels
            return grids, labels
        else:
            # Return grids, labels and affinities
            return grids, labels, affinities

    def __iter__(self):
        """
        Return iterator for a new epoch.

        Notes
        -----
        At every epoch the iterator is reset.

        See https://pytorch.org/ignite/concepts.html#engine for details about the
        internals of :code:`ignite.ending.engine.Engine` and the associated
        abstraction loop.
        """
        # Reset iterator
        self.last_epoch = False
        self.batch_idx = 0
        # TODO: Check that we actually need to reset the iterator
        self.example_provider.reset()
        return self
