"""Simple tests for napari-layer-divider widget."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

# Set headless mode for testing
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_layer_divider._widget import LayerDivider, divide_image_layers_by_z


def test_divide_function_basic():
    """Test basic image division functionality."""
    # Create test data: (T=1, Z=6, Y=4, X=4)
    data = np.ones((1, 6, 4, 4), dtype=np.float32)

    # Divide at Z=2 and Z=4
    result = divide_image_layers_by_z(data, [2, 4], include_boundaries=False)

    # Should create 3 layers
    assert len(result) == 3
    assert all(layer.shape == data.shape for layer in result)


def test_divide_function_with_boundaries():
    """Test division with boundary inclusion."""
    data = np.ones((1, 6, 4, 4), dtype=np.float32)

    result = divide_image_layers_by_z(data, [2], include_boundaries=True)

    assert len(result) == 2


def test_divide_function_errors():
    """Test error conditions."""
    data = np.ones((1, 6, 4, 4), dtype=np.float32)

    # Test invalid shape
    with pytest.raises(ValueError):
        divide_image_layers_by_z(np.ones((6, 4, 4)), [2])

    # Test invalid Z position
    with pytest.raises(ValueError):
        divide_image_layers_by_z(data, [10])  # Z=10 is out of range


def test_widget_initialization():
    """Test widget can be created."""
    # Create mock viewer
    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    # Create widget
    widget = LayerDivider(viewer)

    # Basic checks
    assert widget.viewer == viewer
    assert hasattr(widget, "layer_combo")
    assert hasattr(widget, "z_input")
    assert hasattr(widget, "split_button")


def test_z_position_parsing():
    """Test Z position input parsing."""
    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    widget = LayerDivider(viewer)

    # Test different input formats
    assert widget.parse_z_positions("3, 7, 5") == [3, 5, 7]
    assert widget.parse_z_positions("[3, 7]") == [3, 7]
    assert widget.parse_z_positions("5") == [5]


def test_widget_with_mock_layer():
    """Test widget with a mock image layer."""
    # Create mock layer
    mock_layer = MagicMock()
    mock_layer.name = "test_image"
    mock_layer.data = np.random.rand(2, 8, 16, 16).astype(np.float32)
    mock_layer.visible = True
    mock_layer.colormap = MagicMock()
    mock_layer.colormap.name = "gray"
    mock_layer.opacity = 1.0

    # Create mock viewer with the layer
    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([mock_layer]))
    viewer.add_image = MagicMock()

    # Create widget
    widget = LayerDivider(viewer)

    # Test layer selection
    widget.layer_combo.currentText = MagicMock(return_value="test_image")
    widget.z_input.setText("3")

    # Test split functionality
    widget.split_layer()

    # Verify split was attempted
    assert viewer.add_image.called
    assert not mock_layer.visible  # Original layer should be hidden
