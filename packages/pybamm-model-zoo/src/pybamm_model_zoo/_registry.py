"""Manifest discovery and the model registry.

Manifests are *parsed*, never imported, so a model whose code is broken or whose
dependencies are missing still appears in the registry and reports a clean failure
at :func:`load` time.

Validation here is only the structural minimum needed to key an entry, so one
malformed field fails one model instead of taking the whole registry down.
"""

from __future__ import annotations

import importlib
import keyword
import re
import sys
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pybamm_model_zoo._exceptions import ManifestError, ModelUnavailableError
from pybamm_model_zoo._paths import PACKAGE_ROOT

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

MANIFEST_NAME = "model.toml"
ENTRY_POINT_GROUP = "pybamm_zoo_models"
TIERS = ("community", "core")
DEFAULT_SOLVE_TIME = 3600.0
DEFAULT_KEY_VARIABLES = ("Voltage [V]",)
#: A model folder's name, and the registry key it declares.
SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def usable_identifier(value: str, pattern: re.Pattern[str]) -> bool:
    """Whether ``value`` fits ``pattern`` and can be written in code as itself.

    The patterns describe the shape of an identifier but cannot exclude a
    keyword, and a slug becomes a module name while a name becomes a class name,
    so ``class`` would pass the shape check and render a syntax error.
    """
    return bool(pattern.match(value)) and not keyword.iskeyword(value)


def split_class_path(class_path: str) -> tuple[str, str]:
    """Split a manifest's ``module.path:AttributeName`` into its two halves.

    Returns empty strings for whichever half is absent, so callers can report a
    malformed value rather than having to catch anything.
    """
    module_path, separator, attribute = class_path.partition(":")
    return (module_path, attribute) if separator else ("", "")


@dataclass(frozen=True)
class Maintainer:
    """A person responsible for reviewing changes to a model."""

    name: str
    github: str


@dataclass(frozen=True)
class Dependencies:
    """A model's third-party dependencies, declared as a zoo extra."""

    extra: str | None = None
    packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestSpec:
    """How the contract suite should exercise a model."""

    parameter_set: str | None = None
    solve_time: float = DEFAULT_SOLVE_TIME
    key_variables: tuple[str, ...] = DEFAULT_KEY_VARIABLES
    skip_contract: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ModelEntry:
    """One registered model, as described by its manifest.

    Attributes are read leniently so that a manifest with a bad field still yields
    an entry; ``check_manifest`` is what turns a bad field into a failure.
    """

    slug: str
    name: str
    path: Path
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    external: bool = False

    @property
    def manifest_path(self) -> Path:
        return self.path / MANIFEST_NAME

    @property
    def _model(self) -> dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def title(self) -> str:
        return self._model.get("title", "")

    @property
    def summary(self) -> str:
        return self._model.get("summary", "")

    @property
    def class_path(self) -> str:
        return self._model.get("class", "")

    @property
    def tier(self) -> str:
        return self._model.get("tier", "community")

    @property
    def pybamm_requires(self) -> str:
        return self._model.get("pybamm_requires", "")

    @property
    def added(self) -> str:
        return self._model.get("added", "")

    @property
    def license(self) -> str:
        return self._model.get("license", "")

    @property
    def maintainers(self) -> tuple[Maintainer, ...]:
        return tuple(
            Maintainer(name=entry.get("name", ""), github=entry.get("github", ""))
            for entry in self._model.get("maintainers", [])
            if isinstance(entry, dict)
        )

    @property
    def citation_key(self) -> str:
        return self._model.get("citation", {}).get("key", "")

    @property
    def dependencies(self) -> Dependencies:
        block = self._model.get("dependencies", {})
        return Dependencies(
            extra=block.get("extra") or None,
            packages=tuple(block.get("packages", [])),
        )

    @property
    def tests(self) -> TestSpec:
        block = self._model.get("tests", {})
        key_variables = tuple(block.get("key_variables", DEFAULT_KEY_VARIABLES))
        return TestSpec(
            parameter_set=block.get("parameter_set") or None,
            solve_time=float(block.get("solve_time", DEFAULT_SOLVE_TIME)),
            key_variables=key_variables,
            skip_contract=frozenset(block.get("skip_contract", [])),
        )

    @property
    def module_path(self) -> str:
        return split_class_path(self.class_path)[0]

    @property
    def attribute(self) -> str:
        return split_class_path(self.class_path)[1]

    def admits(self, version: str) -> bool:
        """Whether ``version`` satisfies the manifest's ``pybamm_requires``.

        Raises
        ------
        ManifestError
            If ``pybamm_requires`` is not a valid version specifier.
        """
        # Imported here so the manifest-only paths (the docs generator, the CI
        # matrix) keep working in an environment with nothing but the stdlib.
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import Version

        try:
            specifier = SpecifierSet(self.pybamm_requires)
        except InvalidSpecifier as error:
            raise ManifestError(
                f"{self.manifest_path}: pybamm_requires "
                f"'{self.pybamm_requires}' is not a valid specifier ({error})"
            ) from error
        # A development checkout reports a .devN version, which a bare specifier
        # excludes; the question here is only whether the range is satisfied.
        return specifier.contains(Version(version), prereleases=True)

    def load(self) -> type:
        """Import and return the model class.

        Raises
        ------
        ManifestError
            If the manifest does not declare a parseable ``class``.
        ModelUnavailableError
            If the module cannot be imported or lacks the named attribute.
        """
        if not self.module_path or not self.attribute:
            raise ManifestError(
                f"{self.manifest_path}: 'class' must be 'module.path:AttributeName', "
                f"got {self.class_path!r}"
            )
        try:
            module = importlib.import_module(self.module_path)
        except ImportError as error:
            extra = self.dependencies.extra
            hint = (
                f" It declares the extra '{extra}'; install it with "
                f"`uv sync --extra {extra}`."
                if extra
                else ""
            )
            raise ModelUnavailableError(
                f"'{self.name}' could not be imported.{hint}"
            ) from error
        try:
            return getattr(module, self.attribute)
        except AttributeError as error:
            raise ModelUnavailableError(
                f"'{self.name}' declares {self.class_path!r} but "
                f"'{self.module_path}' has no attribute '{self.attribute}'"
            ) from error


def read_manifest(path: Path) -> dict[str, Any]:
    """Parse a ``model.toml``.

    Raises
    ------
    ManifestError
        If the file is missing or is not valid TOML.
    """
    try:
        with path.open("rb") as manifest:
            return tomllib.load(manifest)
    except FileNotFoundError as error:
        raise ManifestError(f"{path}: no such manifest") from error
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{path}: invalid TOML ({error})") from error


def _entry_from_manifest(path: Path, *, external: bool) -> ModelEntry:
    raw = read_manifest(path)
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ManifestError(f"{path}: missing a [model] table")
    slug = model.get("slug")
    name = model.get("name")
    for label, value in (("slug", slug), ("name", name)):
        if not isinstance(value, str) or not value:
            raise ManifestError(f"{path}: [model].{label} must be a non-empty string")
    return ModelEntry(
        slug=slug, name=name, path=path.parent, raw=raw, external=external
    )


class Registry(Mapping[str, ModelEntry]):
    """Mapping of model name to :class:`ModelEntry`, discovered from manifests."""

    def __init__(
        self,
        paths: list[Path] | None = None,
        *,
        external_paths: list[Path] | None = None,
    ) -> None:
        self._entries: dict[str, ModelEntry] = {}
        self._by_slug: dict[str, ModelEntry] = {}
        for root in [builtin_root()] if paths is None else paths:
            self._discover(root, external=False)
        for root in (
            external_model_paths() if external_paths is None else external_paths
        ):
            self._discover(root, external=True)

    def _shadowed(self, entry: ModelEntry) -> tuple[str, ModelEntry] | None:
        """The registered entry ``entry`` would displace, and how it clashes.

        Both keys have to be checked: ``by_slug`` resolves citations and the
        generated per-model files, so a clashing slug shadows a model just as
        effectively as a clashing name.
        """
        for label, registered in (
            (f"name '{entry.name}'", self._entries.get(entry.name)),
            (f"slug '{entry.slug}'", self._by_slug.get(entry.slug)),
        ):
            if registered is not None:
                return label, registered
        return None

    def _discover(self, root: Path, *, external: bool) -> None:
        for manifest in sorted(Path(root).glob(f"*/{MANIFEST_NAME}")):
            entry = _entry_from_manifest(manifest, external=external)
            clash = self._shadowed(entry)
            if clash is None:
                self._entries[entry.name] = entry
                self._by_slug[entry.slug] = entry
                continue
            label, existing = clash
            if external:
                # In-tree models win, so a third-party package cannot shadow one.
                warnings.warn(
                    f"ignoring external model '{entry.name}' from {manifest}: "
                    f"{label} is already registered by {existing.manifest_path}",
                    stacklevel=2,
                )
            else:
                raise ManifestError(
                    f"{manifest}: duplicate model {label}, already "
                    f"declared by {existing.manifest_path}"
                )

    def __getitem__(self, name: str) -> ModelEntry:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "none"
            raise KeyError(
                f"'{name}' is not a registered model. Registered: {known}."
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def by_slug(self, slug: str) -> ModelEntry:
        """Return the entry whose folder name is ``slug``."""
        try:
            return self._by_slug[slug]
        except KeyError:
            known = ", ".join(sorted(self._by_slug)) or "none"
            raise KeyError(
                f"no registered model with slug '{slug}'. Known: {known}."
            ) from None


def builtin_root() -> Path:
    """The directory holding the in-tree model folders."""
    return PACKAGE_ROOT


def external_model_paths() -> list[Path]:
    """Model directories advertised by third-party packages via the entry point."""
    paths: list[Path] = []
    for entry_point in _iter_entry_points():
        try:
            module = importlib.import_module(entry_point.value)
        except ImportError:  # pragma: no cover - depends on the environment
            warnings.warn(
                f"could not import '{entry_point.value}' advertised by the "
                f"'{ENTRY_POINT_GROUP}' entry point '{entry_point.name}'",
                stacklevel=2,
            )
            continue
        for location in module.__path__:
            paths.append(Path(location))
    return paths


def _iter_entry_points():
    from importlib.metadata import entry_points

    return entry_points(group=ENTRY_POINT_GROUP)
