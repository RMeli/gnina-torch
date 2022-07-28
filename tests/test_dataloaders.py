import pytest
import torch

from gninatorch import setup, training
from gninatorch.dataloaders import GriddedExamplesLoader


@pytest.mark.parametrize("iteration_scheme", ["small", "large"])
def test_GriddedExamplesLoader(trainfile, dataroot, device, iteration_scheme):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "1",
            "--iteration_scheme",
            iteration_scheme,
        ]
    )

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert dataset.num_examples_tot == 3
    assert dataset.num_labels == 3

    # Without balancing the length of the dataset is the same as the number of examples
    # This is true for both small and large epochs
    assert dataset.example_provider.small_epoch_size() == 3
    assert dataset.example_provider.large_epoch_size() == 3

    for _ in range(10):  # Simulate epochs
        # Simulate batches
        # Expect three batches all of the same size
        for _ in range(3):
            grids, labels = next(dataset)
            assert grids.shape == (1, 28, 48, 48, 48)
            assert labels.shape == (1,)

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset)

        # Restart iterator
        dataset = iter(dataset)


@pytest.mark.parametrize("iteration_scheme", ["small", "large"])
def test_GriddedExamplesLoader_batch_size_2l(
    trainfile, dataroot, device, iteration_scheme
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "2",
            "--iteration_scheme",
            iteration_scheme,
        ]
    )

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert dataset.num_examples_tot == 3
    assert dataset.num_labels == 3

    # Without balancing the length of the dataset is the same as the number of examples
    # This is true for both small and large epochs
    assert dataset.example_provider.small_epoch_size() == dataset.num_examples_tot
    assert dataset.example_provider.large_epoch_size() == dataset.num_examples_tot
    assert dataset.num_examples_per_epoch == dataset.num_examples_tot

    assert dataset.num_batches == 2
    assert dataset.last_batch_size == 1

    for epoch in range(10):  # Simulate epochs
        # First batch
        grids, labels = next(dataset)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)

        # Second batch; this batch only contains one example
        grids, labels = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset)

        # Restart iterator
        dataset = iter(dataset)


def test_GriddedExamplesLoader_iteration_scheme_balanced_batch_size_1(
    trainfile, dataroot, device
):
    """
    Notes
    -----
    With :code:`batch_size` set to 1 as here, the sampling can't be balanced since the
    batch only contains a single example.
    """

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--balanced", "--batch_size", "1"]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    with pytest.raises(ValueError):
        GriddedExamplesLoader(
            example_provider=e_small,
            grid_maker=gmaker_small,
            device=device,
        )

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "1",
            "--iteration_scheme",
            "large",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    with pytest.raises(ValueError):
        GriddedExamplesLoader(
            example_provider=e_large,
            grid_maker=gmaker_large,
            device=device,
        )


def test_GriddedExamplesLoader_iteration_scheme_balanced_batch_size_2(
    trainfile, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [trainfile, "-d", dataroot, "--no_shuffle", "--balanced", "--batch_size", "2"]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_small.num_examples_tot == 3
    assert dataset_small.num_labels == 3
    assert dataset_small.num_examples_per_epoch == 2  # Twice the minority class
    assert dataset_small.num_batches == 1
    assert dataset_small.last_batch_size == 0

    # With a batch_size of 2, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    for _ in range(10):  # Simulate epochs
        # Load the only batch
        grids, labels = next(dataset_small)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)

        # The positive example is sampled once, with one of the negative examples
        assert torch.allclose(labels, torch.tensor([1, 0], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_small)

        # Restart iterator
        dataset_small = iter(dataset_small)

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--iteration_scheme",
            "large",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_large.num_examples_tot == 3
    assert dataset_large.num_labels == 3
    assert dataset_large.num_examples_per_epoch == 4  # Twice the majority class
    assert dataset_large.num_batches == 2
    assert dataset_large.last_batch_size == 0

    # With a batch_size of 2, only two batches can be loaded in a large epoch since
    # there are two examples in the manjority class
    for _ in range(10):  # Simulate epochs
        # Load two batches
        for _ in range(2):
            grids, labels = next(dataset_large)
            assert grids.shape == (2, 28, 48, 48, 48)
            assert labels.shape == (2,)

            # The positive example is sampled twice, one for each of the negative examples
            assert torch.allclose(labels, torch.tensor([1, 0], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_large)

        # Restart iterator
        dataset_large = iter(dataset_large)


@pytest.mark.parametrize("iteration_scheme", ["small", "large"])
def test_GriddedExamplesLoader_batch_size_2_no_balancing(
    trainfile, dataroot, device, iteration_scheme
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "2",
            "--iteration_scheme",
            iteration_scheme,
        ]
    )

    e = setup.setup_example_provider(args.trainfile, args)
    gmaker = setup.setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device
    )

    assert dataset.num_examples_tot == 3
    assert dataset.num_examples_per_epoch == 3
    assert dataset.num_labels == 3
    assert dataset.num_batches == 2
    assert dataset.last_batch_size == 1

    for _ in range(10):  # Simulate epochs

        # Load first batch with two examples
        grids, labels = next(dataset)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)
        assert torch.allclose(labels, torch.tensor([0, 1], device=device))

        # Load second (last) batch with only one example
        grids, labels = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)
        assert labels.shape == (1,)
        assert torch.allclose(labels, torch.tensor([0], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset)

        # Restart iterator
        dataset = iter(dataset)


def test_GriddedExamplesLoader_balanced_batch_size_2_affinity(
    trainfile, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--affinity_pos",
            "1",
        ]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        affinity_pos=args_small.affinity_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_small.num_examples_tot == 3
    assert dataset_small.num_labels == 3
    assert dataset_small.num_examples_per_epoch == 2  # Twice the minority class
    assert dataset_small.num_batches == 1
    assert dataset_small.last_batch_size == 0

    # With a batch_size of 2, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    for _ in range(10):  # Simulate epochs
        # Load the only batch
        grids, labels, affinity = next(dataset_small)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)
        assert affinity.shape == (2,)

        # The positive example is sampled once, with one of the negative examples
        assert torch.allclose(labels, torch.tensor([1, 0], device=device))

        assert torch.allclose(affinity, torch.tensor([2.1, 1.1], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_small)

        # Restart iterator
        dataset_small = iter(dataset_small)

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--iteration_scheme",
            "large",
            "--affinity_pos",
            "1",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        affinity_pos=args_large.affinity_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_large.num_examples_tot == 3
    assert dataset_large.num_labels == 3
    assert dataset_large.num_examples_per_epoch == 4  # Twice the majority class
    assert dataset_large.num_batches == 2
    assert dataset_large.last_batch_size == 0

    # With a batch_size of 2, only two batches can be loaded in a large epoch since
    # there are two examples in the manjority class
    for _ in range(10):  # Simulate epochs
        # Load two batches
        for _ in range(2):
            grids, labels, affinity = next(dataset_large)
            assert grids.shape == (2, 28, 48, 48, 48)
            assert labels.shape == (2,)
            assert affinity.shape == (2,)

            # The positive example is sampled twice, one for each of the negative examples
            assert torch.allclose(labels, torch.tensor([1, 0], device=device))

            # Two possible outcomes for the binding affinity
            # with balancing and large epoch
            p1 = torch.tensor([2.1, 1.1], device=device)
            p2 = torch.tensor([2.1, 3.1], device=device)
            assert torch.allclose(affinity, p1) or torch.allclose(affinity, p2)

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_large)

        # Restart iterator
        dataset_large = iter(dataset_large)


def test_GriddedExamplesLoader_balanced_batch_size_2_flexlabel(
    trainfile, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--flexlabel_pos",
            "2",
        ]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        flexlabel_pos=args_small.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_small.num_examples_tot == 3
    assert dataset_small.num_labels == 3
    assert dataset_small.num_examples_per_epoch == 2  # Twice the minority class
    assert dataset_small.num_batches == 1
    assert dataset_small.last_batch_size == 0

    # With a batch_size of 2, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    for _ in range(10):  # Simulate epochs
        # Load the only batch
        grids, labels, flexlabels = next(dataset_small)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)
        assert flexlabels.shape == (2,)

        # The positive example is sampled once, with one of the negative examples
        assert torch.allclose(labels, torch.tensor([1, 0], device=device))

        assert torch.allclose(flexlabels, torch.tensor([1, 1], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_small)

        # Restart iterator
        dataset_small = iter(dataset_small)

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--iteration_scheme",
            "large",
            "--flexlabel_pos",
            "2",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        flexlabel_pos=args_large.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_large.num_examples_tot == 3
    assert dataset_large.num_labels == 3
    assert dataset_large.num_examples_per_epoch == 4  # Twice the majority class
    assert dataset_large.num_batches == 2
    assert dataset_large.last_batch_size == 0

    # With a batch_size of 2, only two batches can be loaded in a large epoch since
    # there are two examples in the manjority class
    for _ in range(10):  # Simulate epochs
        # Load two batches
        for _ in range(2):
            grids, labels, flexlabels = next(dataset_large)
            assert grids.shape == (2, 28, 48, 48, 48)
            assert labels.shape == (2,)
            assert flexlabels.shape == (2,)

            # The positive example is sampled twice, one for each of the negative examples
            assert torch.allclose(labels, torch.tensor([1, 0], device=device))

            # Two possible outcomes for the binding affinity
            # with balancing and large epoch
            p1 = torch.tensor([1, 1], device=device)
            p2 = torch.tensor([1, 0], device=device)
            assert torch.allclose(flexlabels, p1) or torch.allclose(flexlabels, p2)

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_large)

        # Restart iterator
        dataset_large = iter(dataset_large)


def test_GriddedExamplesLoader_balanced_batch_size_2_affinity_flexlabel(
    trainfile, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--affinity_pos",
            "1",
            "--flexlabel_pos",
            "2",
        ]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        affinity_pos=args_small.affinity_pos,
        flexlabel_pos=args_small.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_small.num_examples_tot == 3
    assert dataset_small.num_labels == 3
    assert dataset_small.num_examples_per_epoch == 2  # Twice the minority class
    assert dataset_small.num_batches == 1
    assert dataset_small.last_batch_size == 0

    # With a batch_size of 2, only one batch can be loaded in a small epoch since there
    # is only one example in the minority class
    for _ in range(10):  # Simulate epochs
        # Load the only batch
        grids, labels, affinity, flexlabels = next(dataset_small)
        assert grids.shape == (2, 28, 48, 48, 48)
        assert labels.shape == (2,)
        assert affinity.shape == (2,)
        assert flexlabels.shape == (2,)

        # The positive example is sampled once, with one of the negative examples
        assert torch.allclose(labels, torch.tensor([1, 0], device=device))
        assert torch.allclose(affinity, torch.tensor([2.1, 1.1], device=device))
        assert torch.allclose(flexlabels, torch.tensor([1, 1], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_small)

        # Restart iterator
        dataset_small = iter(dataset_small)

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfile,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "2",
            "--iteration_scheme",
            "large",
            "--affinity_pos",
            "1",
            "--flexlabel_pos",
            "2",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        affinity_pos=args_large.affinity_pos,
        flexlabel_pos=args_large.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_large.num_examples_tot == 3
    assert dataset_large.num_labels == 3
    assert dataset_large.num_examples_per_epoch == 4  # Twice the majority class
    assert dataset_large.num_batches == 2
    assert dataset_large.last_batch_size == 0

    # With a batch_size of 2, only two batches can be loaded in a large epoch since
    # there are two examples in the manjority class
    for _ in range(10):  # Simulate epochs
        # Load two batches
        for _ in range(2):
            grids, labels, affinity, flexlabels = next(dataset_large)
            assert grids.shape == (2, 28, 48, 48, 48)
            assert labels.shape == (2,)
            assert affinity.shape == (2,)
            assert flexlabels.shape == (2,)

            # The positive example is sampled twice, one for each of the negative examples
            assert torch.allclose(labels, torch.tensor([1, 0], device=device))

            # Two possible outcomes for the binding affinity
            # with balancing and large epoch
            b1 = torch.tensor([2.1, 1.1], device=device)
            b2 = torch.tensor([2.1, 3.1], device=device)
            assert torch.allclose(affinity, b1) or torch.allclose(affinity, b2)

            # Two possible outcomes for the flexible residues pose label
            # with balancing and large epoch
            p1 = torch.tensor([1, 1], device=device)
            p2 = torch.tensor([1, 0], device=device)
            assert torch.allclose(flexlabels, p1) or torch.allclose(flexlabels, p2)

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_large)

        # Restart iterator
        dataset_large = iter(dataset_large)


def test_GriddedExamplesLoader_balanced_batch_size_2_flexlabel_stratified(
    trainfilestrat, dataroot, device
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_small = training.options(
        [
            trainfilestrat,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "4",
            "--affinity_pos",
            "1",
            "--flexlabel_pos",
            "2",
            "--stratify_pos",
            "2",
            "--stratify_min",
            "0",
            "--stratify_max",
            "1",
            "--stratify_step",
            "0.5",
        ]
    )
    e_small = setup.setup_example_provider(args_small.trainfile, args_small)
    gmaker_small = setup.setup_grid_maker(args_small)

    dataset_small = GriddedExamplesLoader(
        example_provider=e_small,
        grid_maker=gmaker_small,
        affinity_pos=args_small.affinity_pos,
        flexlabel_pos=args_small.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_small.num_examples_tot == 12
    assert dataset_small.num_labels == 3
    assert dataset_small.last_batch_size == 0

    # Despite having 12 examples, the last 4 have all the same annotation for the flexible pose
    # This means that the last 4 examples are not sampled (i.e. there is one less batch)
    assert dataset_small.num_examples_per_epoch == 8  # Twice the minority class
    assert dataset_small.num_batches == 2

    for _ in range(10):  # Simulate epochs
        for i in range(dataset_small.num_batches):  # Simulate batches
            # Load the only batch
            grids, labels, affinity, flexlabels = next(dataset_small)
            assert grids.shape == (args_small.batch_size, 28, 48, 48, 48)
            assert labels.shape == (args_small.batch_size,)
            assert affinity.shape == (args_small.batch_size,)
            assert flexlabels.shape == (args_small.batch_size,)

            assert torch.allclose(labels, torch.tensor([1, 1, 0, 0], device=device))
            assert torch.allclose(flexlabels, torch.tensor([0, 1, 0, 1], device=device))
            # This reflects the ordering in the file since no shuffling is involved
            if i % 2 == 0:
                assert torch.allclose(
                    affinity,
                    torch.tensor([2.1000, 4.1000, 1.1000, 3.1000], device=device),
                )
            else:
                assert torch.allclose(
                    affinity,
                    torch.tensor([0.2000, 0.4000, 0.1000, 0.3000], device=device),
                )

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_small)

        # Restart iterator
        dataset_small = iter(dataset_small)

    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args_large = training.options(
        [
            trainfilestrat,
            "-d",
            dataroot,
            "--no_shuffle",
            "--balanced",
            "--batch_size",
            "4",
            "--iteration_scheme",
            "large",
            "--affinity_pos",
            "1",
            "--flexlabel_pos",
            "2",
            "--stratify_pos",
            "2",
            "--stratify_min",
            "0",
            "--stratify_max",
            "1",
            "--stratify_step",
            "0.5",
        ]
    )
    e_large = setup.setup_example_provider(args_large.trainfile, args_large)
    gmaker_large = setup.setup_grid_maker(args_large)

    dataset_large = GriddedExamplesLoader(
        example_provider=e_large,
        grid_maker=gmaker_large,
        affinity_pos=args_large.affinity_pos,
        flexlabel_pos=args_large.flexlabel_pos,
        device=device,
    )

    # Dataset test.types contains one positive example and two negative examples
    # Balancing (minority class oversampling) results in different epoch sizes
    assert dataset_large.num_examples_tot == 12
    assert dataset_large.num_labels == 3

    assert dataset_large.num_examples_per_epoch == 16
    assert dataset_large.num_batches == 4

    assert dataset_large.last_batch_size == 0

    for _ in range(10):  # Simulate epochs
        # Load two batches
        for _ in range(dataset_large.num_batches):
            grids, labels, affinity, flexlabels = next(dataset_large)
            assert grids.shape == (args_large.batch_size, 28, 48, 48, 48)
            assert labels.shape == (args_large.batch_size,)
            assert affinity.shape == (args_large.batch_size,)
            assert flexlabels.shape == (args_large.batch_size,)

            print(labels, affinity, flexlabels)

            # Large epoch still has balanced batches
            assert torch.allclose(labels, torch.tensor([1, 1, 0, 0], device=device))
            assert torch.allclose(flexlabels, torch.tensor([0, 1, 0, 1], device=device))

        # Check that the iterator is exhausted at the end of an epoch
        with pytest.raises(StopIteration):
            next(dataset_large)

        # Restart iterator
        dataset_large = iter(dataset_large)


@pytest.mark.parametrize("iteration_scheme", ["small", "large"])
def test_GriddedExamplesLoader_gridonly(
    testfile_nolabels, dataroot, device, iteration_scheme
):
    # Do not shuffle examples randomly when loading the batch
    # This ensures reproducibility
    args = training.options(
        [
            testfile_nolabels,
            "-d",
            dataroot,
            "--no_shuffle",
            "--batch_size",
            "1",
            "--iteration_scheme",
            iteration_scheme,
        ]
    )

    e = setup.setup_example_provider(args.trainfile, args, training=False)
    gmaker = setup.setup_grid_maker(args)

    dataset = GriddedExamplesLoader(
        example_provider=e, grid_maker=gmaker, device=device, grids_only=True
    )

    assert dataset.num_examples_tot == 3
    assert dataset.num_labels == 0

    # Without balancing the length of the dataset is the same as the number of examples
    # This is true for both small and large epochs
    assert dataset.example_provider.small_epoch_size() == 3
    assert dataset.example_provider.large_epoch_size() == 3

    for _ in range(3):
        grids = next(dataset)
        assert grids.shape == (1, 28, 48, 48, 48)

    # Check that the iterator is exhausted at the end of an epoch
    with pytest.raises(StopIteration):
        next(dataset)
