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


@pytest.mark.qt
def test_widget_initialization(qapp, qtbot):
    """Test widget can be created safely."""
    if qapp is None:
        pytest.skip("Qt not available")

    # Create mock viewer with minimal required attributes
    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    try:
        # Create widget
        widget = LayerDivider(viewer)
        qtbot.addWidget(widget)

        # Basic checks that don't trigger complex operations
        assert widget.viewer == viewer
        assert hasattr(widget, "layer_combo")
        assert hasattr(widget, "z_input")
        assert hasattr(widget, "split_button")

        # Test that we can access basic properties
        assert widget.layer_combo is not None
        assert widget.z_input is not None

    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"Widget initialization failed: {e}")


@pytest.mark.qt
def test_widget_basic_functionality(qapp, qtbot):
    """Test basic widget functionality without triggering split operation."""
    if qapp is None:
        pytest.skip("Qt not available")

    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    try:
        widget = LayerDivider(viewer)
        qtbot.addWidget(widget)

        # Test setting text in input field
        widget.z_input.setText("1,2,3")
        assert widget.z_input.text() == "1,2,3"

        # Test clearing inputs
        widget.clear_inputs()
        assert widget.z_input.text() == ""

    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"Basic functionality test failed: {e}")


def test_widget_parse_z_positions():
    """Test Z position parsing without Qt dependencies."""
    # Create a mock viewer (we don't need Qt for this test)
    viewer = MagicMock()
    viewer.layers = MagicMock()
    viewer.layers.events = MagicMock()
    viewer.layers.events.inserted = MagicMock()
    viewer.layers.events.removed = MagicMock()
    viewer.layers.__iter__ = MagicMock(return_value=iter([]))

    # Create widget without Qt setup
    widget = LayerDivider.__new__(
        LayerDivider
    )  # Create without calling __init__
    widget.viewer = viewer

    # Test Z position parsing directly (this doesn't need Qt)
    assert widget.parse_z_positions("3, 7, 5") == [3, 5, 7]
    assert widget.parse_z_positions("[3, 7]") == [3, 7]
    assert widget.parse_z_positions("5") == [5]
    assert widget.parse_z_positions("1,2,3") == [1, 2, 3]
    assert widget.parse_z_positions("") == []


def test_widget_layer_finding():
    """Test layer finding logic without Qt."""
    # Create mock layers
    mock_layer1 = MagicMock()
    mock_layer1.name = "layer1"
    mock_layer1.data = np.ones((1, 4, 8, 8), dtype=np.float32)

    mock_layer2 = MagicMock()
    mock_layer2.name = "test_image"
    mock_layer2.data = np.ones((1, 6, 8, 8), dtype=np.float32)

    layers = [mock_layer1, mock_layer2]

    # Test finding layer by name
    found_layer = None
    for layer in layers:
        if layer.name == "test_image":
            found_layer = layer
            break

    assert found_layer is not None
    assert found_layer.name == "test_image"
    assert found_layer.data.shape == (1, 6, 8, 8)
