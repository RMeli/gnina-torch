import os

import molgrid
import pytest
import torch


def pytest_addoption(parser):
    # Allows user to force tests to run on the CPU (useful to get performance on CI)
    parser.addoption(
        "--nogpu",
        action="store_false",
        help="Force tests to run on CPU",
    )


@pytest.fixture(scope="session")
def device(pytestconfig):
    """
    Configure device.

    Notes
    -----
    Tests run automatically on the GPU if available, unless the user forces tests to run
    ion the CPU by passing the :code:`--nogpu` option.
    """

    gpu = pytestconfig.getoption("--nogpu")

    if gpu:
        device_index = 0
        device = torch.device(
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
        molgrid.set_gpu_device(device_index)
    else:
        device = torch.device("cpu")

    return device


@pytest.fixture(scope="session")
def trainfile() -> str:
    """
    Path to small training file.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "test.types")


@pytest.fixture(scope="session")
def trainfilestrat() -> str:
    """
    Path to small training file which allows stratification by flexible residues.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "teststrat.types")


@pytest.fixture
def testfile() -> str:
    """
    Path to small test file.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "test.types")


@pytest.fixture
def testfile_nolabels() -> str:
    """
    Path to small test file.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "test_nolabels.types")


@pytest.fixture(scope="session")
def dataroot() -> str:
    """
    Path to test directory.
    """
    path = os.path.dirname(__file__)
    return os.path.join(path, "data", "mols")
