"""
conftest.py — pytest configuration for the nuScenes VLM Phase 1 test suite.

Adds src/ to sys.path so that test files can import project modules without
installing the package.
"""
import sys
from pathlib import Path

# Make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
