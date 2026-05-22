"""
conftest.py — shared pytest fixtures and path setup.
"""
import sys
import os

# Make the football_tipster package importable from the tests sub-directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
