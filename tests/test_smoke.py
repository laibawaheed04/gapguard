import numpy as np
import gapguard


def test_package_imports():
    # If this passes, the project is installed correctly.
    assert gapguard is not None
    assert np.__version__