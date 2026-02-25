#!/usr/bin/env python3
"""
Setup script to add MSMR whole-cell manuscript utilities to Python path.
Run this script before importing MSMR modules.
"""

import sys
from pathlib import Path

# Get the path to the MSMR jupyter folder (parent of utilities)
current_dir = Path(__file__).parent
msmr_jupyter_path = current_dir.parent / "msmr-whole-cell-manuscript" / "jupyter"

# Add to Python path if not already there
if str(msmr_jupyter_path) not in sys.path:
    sys.path.insert(0, str(msmr_jupyter_path))
    print(f"Added {msmr_jupyter_path} to Python path")

# Test import
try:
    import utilities.msmr as msmr
    import utilities.plotting as plotting

    print("✓ Successfully imported MSMR modules")
    print("Available modules: utilities.msmr, utilities.plotting")
except ImportError as e:
    print(f"✗ Error importing MSMR modules: {e}")

if __name__ == "__main__":
    print("MSMR utilities setup complete!")
    print("You can now import the modules in your scripts:")
    print("from setup_msmr_imports import *")
    print("import utilities.msmr as msmr")
    print("import utilities.plotting as plotting")
