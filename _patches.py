import sys
from unittest.mock import patch

def fix_torchvision_metadata():
    """Patch importlib.metadata to work with symlinked torchvision."""
    def mock_version(pkg_name):
        if pkg_name == "torchvision":
            return "0.21.0"  # Must match your actual version
        try:
            from importlib.metadata import version
            return version(pkg_name)  # Default behavior for other packages
        except:
            return "0.0.0"  # Fallback

    sys.modules["importlib.metadata"] = patch(
        "importlib.metadata.version", 
        side_effect=mock_version
    ).start()

# Apply the patch immediately when imported
fix_torchvision_metadata()