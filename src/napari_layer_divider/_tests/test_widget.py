from unittest.mock import MagicMock

import napari
import numpy as np
import pytest
from napari.layers import Image

from napari_layer_divider._widget import LayerDivider, divide_image_layers_by_z


class TestDivideImageLayersByZ:
    """Test the divide_image_layers_by_z function"""

    def setup_method(self):
        """Set up test data"""
        # Create a 4D test image (T=2, Z=10, Y=5, X=5)
        self.test_data = np.random.rand(2, 10, 5, 5).astype(np.float32)

    def test_basic_division(self):
        """Test basic division functionality"""
        z_divisions = [3, 7]
        result = divide_image_layers_by_z(
            self.test_data, z_divisions, include_boundaries=False
        )

        # Should create 3 layers (0-3, 3-7, 7-10)
        assert len(result) == 3

        # Check shapes
        for layer in result:
            assert layer.shape == self.test_data.shape

        # Check data integrity - first layer should have data in slices 0-2
        np.testing.assert_allclose(
            result[0][:, :3, :, :], self.test_data[:, :3, :, :]
        )
        np.testing.assert_allclose(
            result[0][:, 3:, :, :], 0
        )  # Rest should be zero

        # Second layer should have data in slices 3-6
        np.testing.assert_allclose(
            result[1][:, 3:7, :, :], self.test_data[:, 3:7, :, :]
        )
        np.testing.assert_allclose(result[1][:, :3, :, :], 0)
        np.testing.assert_allclose(result[1][:, 7:, :, :], 0)

        # Third layer should have data in slices 7-9
        np.testing.assert_allclose(
            result[2][:, 7:, :, :], self.test_data[:, 7:, :, :]
        )
        np.testing.assert_allclose(result[2][:, :7, :, :], 0)

    def test_division_with_boundaries(self):
        """Test division with boundary inclusion"""
        z_divisions = [3, 7]
        result = divide_image_layers_by_z(
            self.test_data, z_divisions, include_boundaries=True
        )

        # Should create 3 layers (0-4, 4-8, 8-10)
        assert len(result) == 3

        # First layer should include boundary slice 3
        np.testing.assert_allclose(
            result[0][:, :4, :, :], self.test_data[:, :4, :, :]
        )
        np.testing.assert_allclose(result[0][:, 4:, :, :], 0)

        # Second layer should start from slice 4 and include boundary slice 7
        np.testing.assert_allclose(
            result[1][:, 4:8, :, :], self.test_data[:, 4:8, :, :]
        )
        np.testing.assert_allclose(result[1][:, :4, :, :], 0)
        np.testing.assert_allclose(result[1][:, 8:, :, :], 0)

    def test_single_division(self):
        """Test with single division point"""
        z_divisions = [5]
        result = divide_image_layers_by_z(
            self.test_data, z_divisions, include_boundaries=False
        )

        assert len(result) == 2

        # First layer: slices 0-4
        np.testing.assert_allclose(
            result[0][:, :5, :, :], self.test_data[:, :5, :, :]
        )
        np.testing.assert_allclose(result[0][:, 5:, :, :], 0)

        # Second layer: slices 5-9
        np.testing.assert_allclose(
            result[1][:, 5:, :, :], self.test_data[:, 5:, :, :]
        )
        np.testing.assert_allclose(result[1][:, :5, :, :], 0)

    def test_unsorted_divisions(self):
        """Test with unsorted division points"""
        z_divisions = [7, 3, 5]  # Unsorted
        result = divide_image_layers_by_z(
            self.test_data, z_divisions, include_boundaries=False
        )

        # Should handle sorting automatically and create 4 layers
        assert len(result) == 4

    def test_invalid_shape_error(self):
        """Test error handling for invalid shape"""
        invalid_data = np.random.rand(10, 5, 5)  # 3D instead of 4D

        with pytest.raises(ValueError, match="Image data must be a 4D array"):
            divide_image_layers_by_z(invalid_data, [3])

    def test_invalid_z_positions_error(self):
        """Test error handling for invalid Z positions"""
        # Z position out of range
        with pytest.raises(
            ValueError, match="Z division positions must be between"
        ):
            divide_image_layers_by_z(self.test_data, [15])  # Z max is 9

        with pytest.raises(
            ValueError, match="Z division positions must be between"
        ):
            divide_image_layers_by_z(self.test_data, [-1])  # Negative

    def test_empty_divisions(self):
        """Test with empty division list"""
        result = divide_image_layers_by_z(
            self.test_data, [], include_boundaries=False
        )

        # Should return single layer with all data
        assert len(result) == 1
        np.testing.assert_allclose(result[0], self.test_data)


class TestLayerDividerWidget:
    """Test the LayerDivider widget"""

    def setup_method(self):
        """Set up test environment"""
        self.viewer = MagicMock(spec=napari.Viewer)
        self.viewer.layers = MagicMock()
        self.viewer.layers.events = MagicMock()
        self.viewer.layers.events.inserted = MagicMock()
        self.viewer.layers.events.removed = MagicMock()

        # Create mock image layer
        self.mock_layer = MagicMock(spec=Image)
        self.mock_layer.name = "test_image"
        self.mock_layer.data = np.random.rand(2, 10, 20, 20)
        self.mock_layer.colormap = MagicMock()
        self.mock_layer.colormap.name = "gray"
        self.mock_layer.opacity = 1.0

        self.viewer.layers.__iter__ = MagicMock(
            return_value=iter([self.mock_layer])
        )

    def test_widget_initialization(self):
        """Test widget initialization"""
        widget = LayerDivider(self.viewer)

        # Check if widget is properly initialized
        assert widget.viewer == self.viewer
        assert hasattr(widget, "layer_combo")
        assert hasattr(widget, "z_input")
        assert hasattr(widget, "include_boundaries_cb")
        assert hasattr(widget, "split_button")

        # Check if event connections are set up
        self.viewer.layers.events.inserted.connect.assert_called_once()
        self.viewer.layers.events.removed.connect.assert_called_once()

    def test_parse_z_positions(self):
        """Test Z position parsing"""
        widget = LayerDivider(self.viewer)

        # Test various input formats
        assert widget.parse_z_positions("3, 7, 5") == [3, 5, 7]
        assert widget.parse_z_positions("[3, 7, 5]") == [3, 5, 7]
        assert widget.parse_z_positions("3,7,5") == [3, 5, 7]
        assert widget.parse_z_positions("5") == [5]
        assert widget.parse_z_positions("5, 5, 3") == [
            3,
            5,
        ]  # Remove duplicates

    def test_z_input_validation(self):
        """Test Z input validation"""
        widget = LayerDivider(self.viewer)

        # Set up mock layer selection
        widget.layer_combo.currentText = MagicMock(return_value="test_image")

        # Test valid input
        widget.z_input.setText("3, 7")
        widget.validate_z_input()
        assert widget.split_button.isEnabled()

        # Test invalid input (out of range)
        widget.z_input.setText("15")  # Out of range for Z=10
        widget.validate_z_input()
        assert not widget.split_button.isEnabled()

    def test_layer_selection_update(self):
        """Test layer selection updates"""
        widget = LayerDivider(self.viewer)

        # Test with no layer selected
        widget.layer_combo.currentText = MagicMock(
            return_value="-- Select Image Layer --"
        )
        widget.on_layer_changed()
        assert not widget.split_button.isEnabled()

        # Test with valid layer selected
        widget.layer_combo.currentText = MagicMock(return_value="test_image")
        widget.on_layer_changed()
        # Should show layer info
        assert "Shape:" in widget.layer_info_label.text()

    def test_clear_inputs(self):
        """Test clearing inputs"""
        widget = LayerDivider(self.viewer)

        # Set some values
        widget.z_input.setText("3, 7")
        widget.include_boundaries_cb.setChecked(True)
        widget.validation_label.setText("Some validation message")
        widget.result_label.setText("Some result message")

        # Clear inputs
        widget.clear_inputs()

        # Check if inputs are cleared
        assert widget.z_input.text() == ""
        assert not widget.include_boundaries_cb.isChecked()
        assert widget.validation_label.text() == ""
        assert widget.result_label.text() == ""

    def test_split_layer_functionality(self):
        """Test the complete split layer functionality"""
        widget = LayerDivider(self.viewer)

        # Set up widget state
        widget.layer_combo.currentText = MagicMock(return_value="test_image")
        widget.z_input.setText("3, 7")
        widget.include_boundaries_cb.setChecked(False)

        # Mock viewer.add_image
        widget.viewer.add_image = MagicMock()

        # Execute split
        widget.split_layer()

        # Check if split layers were added to viewer
        assert (
            widget.viewer.add_image.call_count == 3
        )  # Should create 3 layers

        # Check if original layer was hidden
        assert not self.mock_layer.visible

        # Check if result message was set
        assert "Successfully split" in widget.result_label.text()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_division_at_boundaries(self):
        """Test division at the very boundaries"""
        test_data = np.random.rand(1, 5, 3, 3)

        # Division at first slice
        result = divide_image_layers_by_z(
            test_data, [0], include_boundaries=False
        )
        assert len(result) == 2

        # Division at last slice
        result = divide_image_layers_by_z(
            test_data, [4], include_boundaries=False
        )
        assert len(result) == 2

    def test_multiple_consecutive_divisions(self):
        """Test consecutive division points"""
        test_data = np.random.rand(1, 10, 3, 3)

        # Consecutive divisions
        result = divide_image_layers_by_z(
            test_data, [2, 3, 4], include_boundaries=False
        )
        assert len(result) == 4

        # With boundaries, some layers might be empty
        result = divide_image_layers_by_z(
            test_data, [2, 3, 4], include_boundaries=True
        )
        assert len(result) <= 4  # Some layers might be skipped if empty
