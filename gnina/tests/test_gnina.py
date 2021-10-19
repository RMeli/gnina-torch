import sys

import molgrid

import gnina


def test_gnina_imported():
    assert "gnina" in sys.modules


def test_molgrid_imported():
    assert "molgrid" in sys.modules
