import sys

import molgrid

import gnina


def test_gnina_imported():
    assert gnina.__name__ in sys.modules


def test_molgrid_imported():
    assert molgrid.__name__ in sys.modules
