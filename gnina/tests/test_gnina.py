import sys

import gnina
import molgrid


def test_gnina_imported():
    assert "gnina" in sys.modules

def test_molgrid_imported():
    assert "molgrid" in sys.modules
