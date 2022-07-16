import sys

import molgrid

import gninatorch


def test_gnina_imported():
    assert gninatorch.__name__ in sys.modules


def test_molgrid_imported():
    assert molgrid.__name__ in sys.modules
