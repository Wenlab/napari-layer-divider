"""Simple pytest configuration for napari-layer-divider tests."""

import os
import warnings

# Set Qt platform to offscreen for headless testing
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")

# Suppress Qt and GUI warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*GUI.*")
warnings.filterwarnings("ignore", message=".*Qt.*")
