import numpy as np
import pytest


@pytest.fixture
def assert_is_ndarray():
    """Recursively assert that all items in a structure are numpy arrays."""

    def _assert(obj):
        if isinstance(obj, list | tuple):
            for item in obj:
                _assert(item)
        else:
            assert isinstance(obj, np.ndarray)

    return _assert
