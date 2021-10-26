import math

import torch


class GriddedExamplesLoader:
    def __init__(
        self,
        batch_size,
        example_provider,
        grid_maker,
        random_translation: float = 0.0,
        random_rotation: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        self.batch_size = batch_size
        self.example_provider = example_provider
        self.grid_maker = grid_maker
        self.random_translation = random_translation
        self.random_rotation = random_rotation
        self.device = device

        # TODO: Check that example provider is populated

        self.num_examples = self.example_provider.size()
        self.num_labels = self.example_provider.num_labels()
        self.num_types = self.example_provider.num_types()

        self.dims = grid_maker.grid_dimensions(self.num_types)

        if self.batch_size > self.num_examples:
            raise ValueError(
                f"Batch size {self.batch_size} is larger "
                + f"than number of examples {self.num_examples}"
            )

        self.num_batches = math.ceil(self.num_examples / self.batch_size)
        self.batch_idx = 0

    def __len__(self):
        return self.num_batches

    # TODO: Avoid padding with next epoch?
    def __next__(self):
        """
        Notes
        -----
        By default :code:molgrid: pads the last batch with examples from the next epoch.
        """
        # Raising StopIteration is needed by PyTorch-Ignite's Engine
        #   https://pytorch.org/ignite/concepts.html#engine
        # If this is not present, the epoch_length is determined by len(self)
        if self.batch_idx == self.num_batches:
            self.batch_idx = 0
            raise StopIteration
        else:
            self.batch_idx += 1

        batch = self.example_provider.next_batch(self.batch_size)

        grids = torch.zeros((self.batch_size, *self.dims), device=self.device)
        labels = torch.zeros((self.batch_size,), device=self.device)

        # Compute grids from examples
        self.grid_maker.forward(
            batch,
            grids,
            random_translation=self.random_rotation,
            random_rotation=self.random_rotation,
        )

        # TODO: Generalise to extract other labels as well
        batch.extract_label(0, labels)

        # Convert labels to integers
        # libmolgrid only supports float input
        labels = labels.long()

        return grids, labels

    def __iter__(self):
        return self
