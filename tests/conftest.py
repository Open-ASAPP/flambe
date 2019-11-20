import os
import pytest


@pytest.fixture
def top_level():
    parent = os.path.dirname
    return parent(parent(os.path.abspath(__file__)))
