"""
Ensure .env is loaded whenever the src package is imported.
"""

from src.core import config as _config  # noqa: F401
