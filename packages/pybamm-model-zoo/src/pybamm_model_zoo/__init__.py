"""The PyBaMM model zoo: community- and partner-contributed battery models.

Each model lives in its own folder alongside a declarative ``model.toml``
manifest. The manifests are the single source of truth for the registry, the
contract test suite, the docs pages, and the CI routing.

Examples
--------
>>> import pybamm_model_zoo as zoo
>>> "SPMSeriesResistance" in zoo.list_models()
True
>>> entry = zoo.info("SPMSeriesResistance")
>>> entry.tier
'core'
"""

from __future__ import annotations

from pathlib import Path

from pybamm_model_zoo._citations import read_citations
from pybamm_model_zoo._exceptions import (
    ManifestError,
    ModelUnavailableError,
    ZooError,
)
from pybamm_model_zoo._registry import (
    ENTRY_POINT_GROUP,
    TIERS,
    ModelEntry,
    Registry,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "TIERS",
    "ManifestError",
    "ModelEntry",
    "ModelUnavailableError",
    "Registry",
    "ZooError",
    "all_entries",
    "info",
    "list_models",
    "load",
    "read_citations",
    "refresh",
    "register_citation",
    "registry",
]

_registry: Registry | None = None


def registry() -> Registry:
    """Return the model registry, building it on first use."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry


def refresh(
    paths: list[Path] | None = None, *, external_paths: list[Path] | None = None
) -> Registry:
    """Rebuild the registry, optionally from explicit search paths.

    Parameters
    ----------
    paths : list of Path, optional
        Directories of model folders held to the in-tree contract. Defaults to the
        zoo's own model directory.
    external_paths : list of Path, optional
        Directories of third-party model folders. Defaults to those advertised
        through the ``pybamm_zoo_models`` entry point.
    """
    global _registry
    _registry = Registry(paths, external_paths=external_paths)
    return _registry


def list_models() -> list[str]:
    """The names of every registered model, sorted."""
    return sorted(registry())


def all_entries() -> list[ModelEntry]:
    """Every registered entry, sorted by slug."""
    return sorted(registry().values(), key=lambda entry: entry.slug)


def info(name: str) -> ModelEntry:
    """Return the manifest-derived metadata for a registered model."""
    return registry()[name]


def load(name: str) -> type:
    """Import and return a registered model class.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    ModelUnavailableError
        If the model's code or its declared extra is unavailable.
    """
    return registry()[name].load()


def register_citation(slug: str, *keys: str) -> None:
    """Credit a zoo model's references through :func:`pybamm.print_citations`.

    Call this from a model's ``__init__`` so that using the model cites its
    author. With no ``keys``, the manifest's ``citation.key`` is registered.

    Parameters
    ----------
    slug : str
        The model's folder name.
    *keys : str
        Citation keys to register from the folder's ``CITATION.bib``. Defaults to
        the manifest's ``citation.key``.

    Raises
    ------
    ManifestError
        If the folder has no ``CITATION.bib`` or a key is not in it.
    """
    import pybamm

    entry = registry().by_slug(slug)
    citations = read_citations(entry.path)
    for key in keys or (entry.citation_key,):
        if key not in citations:
            raise ManifestError(
                f"{entry.path / 'CITATION.bib'}: no entry for '{key}'. "
                f"Found: {', '.join(sorted(citations)) or 'none'}."
            )
        pybamm.citations.register(citations[key])
