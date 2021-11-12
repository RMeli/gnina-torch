from typing import Optional

import molgrid
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

        self.num_examples = self.example_provider.size()
        self.num_labels = self.example_provider.num_labels()
        self.num_types = self.example_provider.num_types()

        self.dims = grid_maker.grid_dimensions(self.num_types)

    # TODO: Check this is what we want
    # TODO: By default, epoch length is defined by len(data)
    # TODO: If data is a finite data iterator with unknown length epoch_length
    # TODO: will be automatically determined when data iterator is exhausted.
    def __len__(self):
        """
        Return length of the epoch (number of examples).

        Notes
        -----
        The number of examples per epoch depends on the :code:`molgrid.IterationScheme`
        used. Without balancing nor stratification, the number of examples per epoch is
        the same for :code:`molgrid.IterationScheme.SmalleEpoch` and
        :code:`molgrid.IterationScheme.LargeEpoch`. For balanced sampling, which sample
        the same number of positive and negative examples, the number of examples in a
        small epoch (examples seen at most once) is twice the size of the minority class
        while for a large epoch (examples seen at least once) it is twice the size of
        the majority class.
        """
        settings = self.example_provider.settings()

        if settings.iteration_scheme == molgrid.IterationScheme.SmallEpoch:
            return self.example_provider.small_epoch_size()
        elif settings.iteration_scheme == molgrid.IterationScheme.LargeEpoch:
            return self.example_provider.large_epoch_size()
        else:
            raise ValueError("Unknown iteration scheme {settings.iteration_scheme}.")

    # TODO: Avoid padding with next epoch?
    # TODO: Does this happen with the current iteration scheme?)
    def __next__(self):
        """
        Get next batch of gridded examples and corresponding labels.

        Notes
        -----
        The batch size is defined in the :code:`example_provider` as
        :code:`default_batch_size`. The number of batches actually depend on the
        :code:`molgrid.IterationScheme` used, also defined in the
        :code:`example_provider`.

        If :code:`molgrid.IterationScheme.SmallEpoch` is used, examples are seen at most
        once. If :code:`molgrid.IterationScheme.LargeEpoch` is used, examples are seen
        at least once.
        """
        # Use pre-defined molgrid.IterationScheme
        batch = next(self.example_provider)

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
        return self
