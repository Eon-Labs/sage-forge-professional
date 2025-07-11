"""Basic tests for the nautilus_test package."""

import nautilus_test


def test_package_version():
    """Test that the package version is accessible."""
    assert hasattr(nautilus_test, "__version__")
    assert nautilus_test.__version__ == "0.1.0"
