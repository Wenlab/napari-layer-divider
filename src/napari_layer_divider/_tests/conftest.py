"""Pytest configuration for napari-layer-divider tests."""

import os
from unittest.mock import MagicMock

import pytest

# Set Qt platform to offscreen for headless testing
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

# Suppress Qt warnings
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="qtpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qtpy")


@pytest.fixture
def make_napari_viewer():
    """Create a napari viewer for testing."""

    def _make_viewer(*args, **kwargs):
        # For CI/headless testing, use a mock viewer
        if (
            os.environ.get("CI")
            or os.environ.get("QT_QPA_PLATFORM") == "offscreen"
        ):
            viewer = MagicMock()
            viewer.layers = MagicMock()
            viewer.layers.events = MagicMock()
            viewer.layers.events.inserted = MagicMock()
            viewer.layers.events.removed = MagicMock()
            viewer.layers.__iter__ = MagicMock(return_value=iter([]))
            viewer.add_image = MagicMock()
            return viewer
        else:
            # For local testing with GUI
            import napari

            return napari.Viewer(*args, **kwargs)

    return _make_viewer


@pytest.fixture
def mock_image_layer():
    """Create a mock image layer for testing."""
    from unittest.mock import MagicMock

    from napari.layers import Image

    layer = MagicMock(spec=Image)
    layer.name = "test_image"
    layer.data = (
        pytest.test_data if hasattr(pytest, "test_data") else MagicMock()
    )
    layer.colormap = MagicMock()
    layer.colormap.name = "gray"
    layer.opacity = 1.0
    layer.visible = True
    return layer
