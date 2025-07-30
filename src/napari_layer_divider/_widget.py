"""
This module containsdef divide_image_layers_by_z(
    image_data: np.ndarray,
    z_divisions: list[int],
    include_boundaries: bool = False
) -> list[np.ndarray]:dget for dividing napari image layers along the Z-axis.
It allows users to specify Z slice positions and whether to include boundary slices in the previous layer.
"""

import napari
import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def divide_image_layers_by_z(
    image_data: np.ndarray,
    z_divisions: list[int],
    include_boundaries: bool = False,
) -> list[np.ndarray]:
    """
    Divide napari image layers with shape (T,Z,Y,X) into multiple layers based on specified Z slice positions

    Parameters:
    -----------
    image_data : np.ndarray
        Image data with shape (T,Z,Y,X)
    z_divisions : list[int]
        List of Z-axis division positions, e.g., [z0, z1, z2, ...]
    include_boundaries : bool, default=False
        Whether to include boundary slices in the previous layer

    Returns:
    --------
    list[np.ndarray]
        List of divided image layers, each element is an array with shape (T,Z,Y,X)
    """

    if len(image_data.shape) != 4:
        raise ValueError("Image data must be a 4D array with shape (T,Z,Y,X)")

    T, Z, Y, X = image_data.shape

    # Validate division positions
    z_divisions = sorted(z_divisions)
    if any(z < 0 or z >= Z for z in z_divisions):
        raise ValueError(f"Z division positions must be between 0 and {Z-1}")

    # Create division ranges
    z_ranges = []
    start_z = 0

    for z_div in z_divisions:
        if include_boundaries:
            z_ranges.append(
                (start_z, z_div + 1)
            )  # boundary is included in the former layer
            start_z = z_div + 1
        else:
            z_ranges.append(
                (start_z, z_div)
            )  # boundary is included in the latter layer
            start_z = z_div

    # Add the last range
    z_ranges.append((start_z, Z))

    # Create divided image layers
    divided_layers = []

    for start_z, end_z in z_ranges:
        if start_z >= end_z:
            continue

        # Initialize zero array
        layer_shape = (T, Z, Y, X)
        new_layer = np.zeros(layer_shape, dtype=image_data.dtype)

        # Fill corresponding Z slices
        new_layer[:, start_z:end_z, :, :] = image_data[:, start_z:end_z, :, :]

        divided_layers.append(new_layer)

    return divided_layers


class LayerDivider(QWidget):
    """Main Widget for Z-axis splitting plugin"""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.setup_ui()

        # Listen to layer changes
        self.viewer.layers.events.inserted.connect(self.update_layer_choices)
        self.viewer.layers.events.removed.connect(self.update_layer_choices)

    def setup_ui(self):
        """Set up user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title_label = QLabel("Z-Axis Layer Splitter")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-weight: bold; font-size: 14px; margin: 10px;"
        )
        layout.addWidget(title_label)

        # Layer selection group
        layer_group = QGroupBox("Select Image Layer")
        layer_layout = QVBoxLayout()
        layer_group.setLayout(layer_layout)

        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        layer_layout.addWidget(self.layer_combo)

        # Layer info display
        self.layer_info_label = QLabel("Please select an image layer")
        self.layer_info_label.setStyleSheet("color: gray; font-size: 11px;")
        layer_layout.addWidget(self.layer_info_label)

        layout.addWidget(layer_group)

        # Z division position input group
        z_group = QGroupBox("Split Settings")
        z_layout = QVBoxLayout()
        z_group.setLayout(z_layout)

        # Z position input
        z_input_layout = QHBoxLayout()
        z_input_layout.addWidget(QLabel("Z slice positions:"))

        self.z_input = QLineEdit()
        self.z_input.setPlaceholderText("e.g., 4, 10 or [4, 10]")
        self.z_input.textChanged.connect(self.validate_z_input)
        z_input_layout.addWidget(self.z_input)

        z_layout.addLayout(z_input_layout)

        # Boundary option
        self.include_boundaries_cb = QCheckBox(
            "Include boundary slices in previous layer"
        )
        self.include_boundaries_cb.setToolTip(
            "Checked: boundary slices included in previous layer\n"
            "Unchecked: boundary slices included in next layer"
        )
        z_layout.addWidget(self.include_boundaries_cb)

        # Input validation hint
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red; font-size: 10px;")
        z_layout.addWidget(self.validation_label)

        layout.addWidget(z_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.split_button = QPushButton("Split Layer")
        self.split_button.clicked.connect(self.split_layer)
        self.split_button.setEnabled(False)
        button_layout.addWidget(self.split_button)

        self.clear_button = QPushButton("Clear Input")
        self.clear_button.clicked.connect(self.clear_inputs)
        button_layout.addWidget(self.clear_button)

        # Add refresh button for fixing blending issues
        self.refresh_button = QPushButton("Fix Blending")
        self.refresh_button.clicked.connect(
            self.fix_layer_blending_after_split
        )
        self.refresh_button.setToolTip(
            "Click if label layers have blending issues after splitting"
        )
        button_layout.addWidget(self.refresh_button)

        layout.addLayout(button_layout)

        # Result information
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("color: green; font-size: 11px;")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        # Initialize layer selection
        self.update_layer_choices()

    def update_layer_choices(self):
        """Update layer selection dropdown"""
        self.layer_combo.clear()

        # Add empty option
        self.layer_combo.addItem("-- Select Image Layer --")

        # Add all image layers
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and len(layer.data.shape) == 4:
                self.layer_combo.addItem(layer.name)

        self.on_layer_changed()

    def on_layer_changed(self):
        """Update information when selected layer changes"""
        layer_name = self.layer_combo.currentText()

        if layer_name == "-- Select Image Layer --" or not layer_name:
            self.layer_info_label.setText("Please select an image layer")
            self.split_button.setEnabled(False)
            return

        # Get selected layer
        layer = None
        for viewer_layer in self.viewer.layers:
            if viewer_layer.name == layer_name:
                layer = viewer_layer
                break

        if layer is None or not isinstance(layer, Image):
            self.layer_info_label.setText("Invalid image layer")
            self.split_button.setEnabled(False)
            return

        # Display layer information
        shape = layer.data.shape
        if len(shape) == 4:
            T, Z, Y, X = shape
            self.layer_info_label.setText(
                f"Shape: {shape} (T={T}, Z={Z}, Y={Y}, X={X})"
            )
            self.validate_z_input()
        else:
            self.layer_info_label.setText(
                f"Unsupported shape: {shape} (requires 4D: T,Z,Y,X)"
            )
            self.split_button.setEnabled(False)

    def validate_z_input(self):
        """Validate Z position input"""
        layer_name = self.layer_combo.currentText()
        z_text = self.z_input.text().strip()

        # Clear previous validation info
        self.validation_label.setText("")
        self.split_button.setEnabled(False)

        if layer_name == "-- Select Image Layer --" or not layer_name:
            return

        if not z_text:
            return

        # Get Z dimension of current layer
        layer = None
        for viewer_layer in self.viewer.layers:
            if viewer_layer.name == layer_name:
                layer = viewer_layer
                break

        if layer is None or len(layer.data.shape) != 4:
            return

        Z = layer.data.shape[1]

        try:
            # Parse Z positions
            z_positions = self.parse_z_positions(z_text)

            # Validate Z positions
            if not z_positions:
                self.validation_label.setText(
                    "Please enter at least one Z position"
                )
                return

            if any(z < 0 or z >= Z for z in z_positions):
                self.validation_label.setText(
                    f"Z positions must be between 0 and {Z-1}"
                )
                return

            # Validation passed
            self.validation_label.setText(
                f"✓ Will split into {len(z_positions)+1} layers"
            )
            self.validation_label.setStyleSheet(
                "color: green; font-size: 10px;"
            )
            self.split_button.setEnabled(True)

        except (ValueError, TypeError) as e:
            self.validation_label.setText(f"Input format error: {str(e)}")

    def parse_z_positions(self, z_text: str) -> list[int]:
        """Parse Z position input"""
        z_text = z_text.strip()

        # Remove brackets
        z_text = z_text.strip("[]")

        # Split by comma
        parts = [part.strip() for part in z_text.split(",")]

        z_positions = []
        for part in parts:
            if part:
                z_positions.append(int(part))

        return sorted(set(z_positions))  # Remove duplicates and sort

    def split_layer(self):
        """Execute layer splitting"""
        try:
            # Get selected layer
            layer_name = self.layer_combo.currentText()
            layer = None
            for viewer_layer in self.viewer.layers:
                if viewer_layer.name == layer_name:
                    layer = viewer_layer
                    break

            if layer is None:
                QMessageBox.warning(
                    self, "Error", "Please select a valid image layer"
                )
                return

            # Parse Z positions
            z_text = self.z_input.text().strip()
            z_positions = self.parse_z_positions(z_text)

            # Execute splitting
            include_boundaries = self.include_boundaries_cb.isChecked()
            divided_layers = divide_image_layers_by_z(
                layer.data, z_positions, include_boundaries
            )

            # Get the original layer's index to maintain proper layer order
            original_layer_index = list(self.viewer.layers).index(layer)

            # Add split layers to viewer at the same position as original layer
            for i, divided_layer in enumerate(divided_layers):
                new_layer_name = f"{layer_name}_split_{i+1}"

                # Prepare layer properties to preserve
                layer_kwargs = {
                    "name": new_layer_name,
                    "opacity": layer.opacity,
                }

                # Preserve colormap
                if hasattr(layer, "colormap"):
                    layer_kwargs["colormap"] = layer.colormap.name
                else:
                    layer_kwargs["colormap"] = "gray"

                # Preserve scale
                if hasattr(layer, "scale"):
                    layer_kwargs["scale"] = layer.scale

                # Preserve translate
                if hasattr(layer, "translate"):
                    layer_kwargs["translate"] = layer.translate

                # Preserve contrast_limits
                if hasattr(layer, "contrast_limits"):
                    layer_kwargs["contrast_limits"] = layer.contrast_limits

                # Preserve blending
                if hasattr(layer, "blending"):
                    layer_kwargs["blending"] = layer.blending

                # Add the new layer
                self.viewer.add_image(divided_layer, **layer_kwargs)

                # Move the new layer to maintain proper order
                # Insert right after the original layer position
                target_index = original_layer_index + i + 1
                if target_index < len(self.viewer.layers):
                    self.viewer.layers.move(
                        len(self.viewer.layers) - 1, target_index
                    )

            # Display result
            self.result_label.setText(
                f"✓ Successfully split into {len(divided_layers)} layers: "
                + ", ".join(
                    [
                        f"{layer_name}_split_{i+1}"
                        for i in range(len(divided_layers))
                    ]
                )
            )

            # Hide original layer (don't remove it to preserve layer relationships)
            layer.visible = False

            # Fix layer blending issues if necessary
            self.fix_layer_blending_after_split()

        except (ValueError, TypeError, AttributeError) as e:
            QMessageBox.critical(
                self,
                "Split Failed",
                f"Error occurred during splitting:\n{str(e)}",
            )

    def fix_layer_blending_after_split(self):
        """
        Helper method to fix layer blending issues after splitting.
        Call this if label layers or other layers have blending problems after division.
        """
        try:
            # Refresh the viewer to update layer interactions
            self.viewer.reset_view()

            # Force a redraw of all layers
            for layer in self.viewer.layers:
                if hasattr(layer, "refresh"):
                    layer.refresh()

        except (ValueError, TypeError, AttributeError) as e:
            print(f"Error fixing layer blending: {e}")

    def clear_inputs(self):
        """Clear inputs"""
        self.z_input.clear()
        self.include_boundaries_cb.setChecked(False)
        self.validation_label.setText("")
        self.result_label.setText("")
