"""
Unit and regression test for the gnina package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import gnina


def test_gnina_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "gnina" in sys.modules
