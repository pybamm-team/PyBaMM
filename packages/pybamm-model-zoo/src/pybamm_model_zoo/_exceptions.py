"""Exceptions raised by the PyBaMM model zoo."""

from __future__ import annotations


class ZooError(Exception):
    """Base class for every model zoo error."""


class ManifestError(ZooError):
    """A ``model.toml`` is missing, unparseable, or fails validation."""


class ModelUnavailableError(ZooError):
    """A registered model exists but its code could not be imported."""
