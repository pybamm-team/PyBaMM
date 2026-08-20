"""The contract every model zoo entry must satisfy.

:data:`CHECKS` is the single definition of that contract: the test suite, the
manifest's ``skip_contract`` validation, and the documentation table all derive
from it, so adding a check is one edit rather than four.

Every check carries a scope, which is what lets a third-party collection be held
to the portable ``MODEL`` rules alone. Nothing here imports ``pytest``, so an
external runner can drive them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from packaging.requirements import Requirement
from packaging.version import Version

import pybamm
from pybamm_model_zoo import _compat, _paths
from pybamm_model_zoo._citations import CITATION_FILE, read_citations
from pybamm_model_zoo._exceptions import ManifestError
from pybamm_model_zoo._registry import (
    NAME_PATTERN,
    SLUG_PATTERN,
    TIERS,
    ModelEntry,
    read_manifest,
    split_class_path,
)

#: A portable rule any zoo model must satisfy, wherever it lives.
MODEL = "model"
#: How an in-tree model is wired into this package.
PACKAGING = "packaging"
#: This repository's own hygiene — a documentation page, an owner.
REPO = "repo"

#: Keys a ``[model]`` table may carry. Anything else is a typo.
MODEL_KEYS = frozenset(
    {
        "slug",
        "name",
        "title",
        "summary",
        "class",
        "tier",
        "pybamm_requires",
        "added",
        "license",
        "maintainers",
        "citation",
        "dependencies",
        "tests",
    }
)
#: README headings the docs pages and the contributor guide both rely on.
REQUIRED_README_SECTIONS = ("Summary", "Usage", "Validation", "Citation")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Check:
    """One contract check, and what it takes to run one.

    Attributes
    ----------
    name : str
        The name a manifest waives it by, and the test id it appears under.
    run : callable
        Takes the entry, raises on failure.
    scope : str
        ``MODEL``, ``PACKAGING``, or ``REPO`` — see the module docstring.
    needs_model : bool
        Whether it imports the model, so it must be skipped when the model's
        declared extra is not installed.
    waivable : bool
        Whether a manifest's ``skip_contract`` may waive it.
    """

    name: str
    run: Callable[[ModelEntry], None]
    scope: str = MODEL
    needs_model: bool = False
    waivable: bool = True


#: The contract, in the order it is worth reading failures in.
CHECKS: dict[str, Check] = {}


def _register(name: str, **attributes: Any) -> Callable[[Callable], Callable]:
    def decorate(function: Callable[[ModelEntry], None]):
        CHECKS[name] = Check(name=name, run=function, **attributes)
        return function

    return decorate


def checks_in_scope(*scopes: str) -> list[Check]:
    """Every check belonging to one of ``scopes``, in registration order."""
    return [check for check in CHECKS.values() if check.scope in scopes]


@_register("manifest", waivable=False)
def check_manifest(entry: ModelEntry) -> None:
    """Every field of the manifest is present, well formed, and consistent."""
    where = entry.manifest_path
    model = entry.raw.get("model", {})

    if unknown := sorted(set(model) - MODEL_KEYS):
        raise AssertionError(f"{where}: unknown [model] key(s) {unknown}")

    assert SLUG_PATTERN.match(entry.slug), (
        f"{where}: slug '{entry.slug}' must be lower_snake_case"
    )
    assert entry.slug == entry.path.name, (
        f"{where}: slug '{entry.slug}' must equal the folder name '{entry.path.name}'"
    )
    assert NAME_PATTERN.match(entry.name), (
        f"{where}: name '{entry.name}' must be a valid Python identifier"
    )
    for label, value in (("title", entry.title), ("summary", entry.summary)):
        assert value.strip(), f"{where}: {label} must be a non-empty string"

    module_path, attribute = split_class_path(entry.class_path)
    assert module_path and attribute, (
        f"{where}: class must be 'module.path:AttributeName', got {entry.class_path!r}"
    )

    assert entry.tier in TIERS, (
        f"{where}: tier must be one of {list(TIERS)}, got '{entry.tier}'"
    )
    assert _DATE_PATTERN.match(entry.added), (
        f"{where}: added must be an ISO date (YYYY-MM-DD), got '{entry.added}'"
    )
    assert entry.license.strip(), f"{where}: license must be an SPDX identifier"

    assert entry.maintainers, f"{where}: at least one [[model.maintainers]] is required"
    for maintainer in entry.maintainers:
        assert maintainer.name.strip(), f"{where}: a maintainer is missing a name"
        assert maintainer.github.strip(), (
            f"{where}: maintainer '{maintainer.name}' is missing a github handle"
        )

    assert entry.citation_key.strip(), f"{where}: [model.citation] key is required"

    check_pybamm_requires(entry)
    _check_tests_block(entry)
    _check_dependency_declaration(entry)


def check_pybamm_requires(entry: ModelEntry) -> None:
    """The declared PyBaMM range is a valid specifier, satisfied by this install.

    Applicability is a *selection* decision elsewhere — the weekly compatibility
    matrix only pairs a model with versions its range admits — so reaching this
    check with an unsatisfied range means the manifest claims a PyBaMM the tests
    are not running against, which is a manifest error worth failing loudly.
    """
    where = entry.manifest_path
    assert entry.pybamm_requires.strip(), (
        f"{where}: pybamm_requires is required, e.g. '>=26.8'"
    )
    try:
        satisfied = entry.admits(pybamm.__version__)
    except ManifestError as error:
        raise AssertionError(str(error)) from error
    assert satisfied, (
        f"{where}: pybamm_requires '{entry.pybamm_requires}' is not satisfied by "
        f"the installed PyBaMM {pybamm.__version__}"
    )


def _check_tests_block(entry: ModelEntry) -> None:
    where = entry.manifest_path
    tests = entry.tests
    assert tests.solve_time > 0, f"{where}: [model.tests] solve_time must be positive"
    assert tests.key_variables, (
        f"{where}: [model.tests] key_variables must name at least one variable"
    )
    if unknown := sorted(tests.skip_contract - set(CHECKS)):
        raise AssertionError(
            f"{where}: skip_contract names unknown check(s) {unknown}; "
            f"valid checks are {list(CHECKS)}"
        )
    if unwaivable := sorted(
        name for name in tests.skip_contract if not CHECKS[name].waivable
    ):
        raise AssertionError(f"{where}: check(s) {unwaivable} cannot be waived")
    if tests.parameter_set:
        assert tests.parameter_set in pybamm.parameter_sets, (
            f"{where}: parameter_set '{tests.parameter_set}' is not a registered "
            f"PyBaMM parameter set"
        )


def _check_dependency_declaration(entry: ModelEntry) -> None:
    """The manifest's own dependency block is coherent."""
    where = entry.manifest_path
    dependencies = entry.dependencies
    if dependencies.packages:
        assert dependencies.extra, (
            f"{where}: [model.dependencies] declares packages but no extra; "
            f"third-party dependencies must be installable as a zoo extra"
        )
    if not dependencies.extra:
        return
    assert dependencies.extra == expected_extra(entry.slug), (
        f"{where}: extra must be named '{expected_extra(entry.slug)}', "
        f"got '{dependencies.extra}'"
    )
    for requirement in dependencies.packages:
        Requirement(requirement)  # raises InvalidRequirement on a malformed pin


def expected_extra(slug: str) -> str:
    """The name of the zoo extra a model's dependencies must be installable by."""
    return f"zoo-{slug.replace('_', '-')}"


@_register("layout")
def check_layout(entry: ModelEntry) -> None:
    """Required files are present and the README carries the required sections."""
    for name in ("README.md", CITATION_FILE):
        assert (entry.path / name).is_file(), f"{entry.path / name}: missing"
    for name in ("examples", "tests"):
        directory = entry.path / name
        assert directory.is_dir(), f"{directory}: missing"
        assert any(directory.glob("**/*.py")), f"{directory}: contains no Python files"

    readme = (entry.path / "README.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^#+\s*(.+?)\s*$", readme, flags=re.MULTILINE))
    missing = [
        section for section in REQUIRED_README_SECTIONS if section not in headings
    ]
    assert not missing, (
        f"{entry.path / 'README.md'}: missing section heading(s) {missing}"
    )


@_register("import", needs_model=True)
def check_import(entry: ModelEntry) -> type[pybamm.BaseModel]:
    """The declared class imports and is a PyBaMM model."""
    model_class = entry.load()
    assert isinstance(model_class, type) and issubclass(
        model_class, pybamm.BaseModel
    ), f"{entry.name}: {entry.class_path} is not a pybamm.BaseModel subclass"
    return model_class


@_register("citation", needs_model=True)
def check_citation(entry: ModelEntry) -> None:
    """The manifest's key resolves, and instantiating the model credits it."""
    citations = read_citations(entry.path)
    assert entry.citation_key in citations, (
        f"{entry.path / CITATION_FILE}: no entry for '{entry.citation_key}'; "
        f"found {sorted(citations)}"
    )
    _compat.reset_citations()
    instantiate(entry)
    assert entry.citation_key in _compat.cited_keys(), (
        f"{entry.name}: instantiating the model does not register "
        f"'{entry.citation_key}', so pybamm.print_citations() will not credit its "
        f"author. Call pybamm_model_zoo.register_citation('{entry.slug}') in "
        f"__init__."
    )


@_register("well_posed", needs_model=True)
def check_well_posed(entry: ModelEntry) -> None:
    """The model's equations form a well-posed system."""
    instantiate(entry).check_well_posedness()


@_register("build", needs_model=True)
def check_build(entry: ModelEntry) -> None:
    """Parameters, geometry, mesh, and discretisation all process the model."""
    _simulation_for(entry).build()


@_register("solve", needs_model=True)
def check_solve(entry: ModelEntry) -> pybamm.Solution:
    """The model solves, and its key variables are finite throughout."""
    solution = _simulation_for(entry).solve([0, entry.tests.solve_time])
    for name in entry.tests.key_variables:
        assert name in solution.all_models[0].variables, (
            f"{entry.name}: key variable '{name}' is not a model variable"
        )
        # Read through the interpolating call interface, not the raw arrays.
        values = np.asarray(solution[name](solution.t))
        assert values.size, f"{entry.name}: '{name}' returned no values"
        assert np.all(np.isfinite(values)), (
            f"{entry.name}: '{name}' is not finite everywhere"
        )
    return solution


@_register("packaging", scope=PACKAGING)
def check_packaging(entry: ModelEntry) -> None:
    """An in-tree model is importable as part of the zoo, with its extra declared."""
    assert (entry.path / "__init__.py").is_file(), (
        f"{entry.path / '__init__.py'}: missing, so the folder is not importable"
    )
    expected_module = f"pybamm_model_zoo.{entry.slug}"
    module_path = entry.module_path
    assert module_path == expected_module or module_path.startswith(
        f"{expected_module}."
    ), (
        f"{entry.manifest_path}: class must live under '{expected_module}', "
        f"got '{module_path}'"
    )
    _check_extra_is_declared(entry)


def _check_extra_is_declared(entry: ModelEntry) -> None:
    """The zoo's pyproject offers the extra the manifest declares, via ``zoo-all``."""
    extra = entry.dependencies.extra
    if not extra:
        return
    pyproject = _paths.ZOO_PYPROJECT
    extras = (
        read_manifest(pyproject).get("project", {}).get("optional-dependencies", {})
    )
    assert extra in extras, (
        f"{pyproject}: no '{extra}' extra, but {entry.manifest_path} declares one"
    )
    declared = {Requirement(item).name for item in extras[extra]}
    missing = sorted(
        {Requirement(item).name for item in entry.dependencies.packages} - declared
    )
    assert not missing, (
        f"{pyproject}: extra '{extra}' is missing {missing}, declared by "
        f"{entry.manifest_path}"
    )
    aggregated: set[str] = set()
    for item in extras.get("zoo-all", []):
        requirement = Requirement(item)
        if requirement.name == "pybamm-model-zoo":
            aggregated |= requirement.extras
    assert extra in aggregated, (
        f"{pyproject}: the 'zoo-all' extra must include 'pybamm-model-zoo[{extra}]', "
        f"or `uv sync --extra zoo-all` will not install {entry.slug}"
    )


@_register("docs", scope=REPO)
def check_docs(entry: ModelEntry) -> None:
    """The generated docs page for the model is present and current."""
    from pybamm_model_zoo import _docs

    for path, content in _docs.pages_for(entry).items():
        assert path.is_file(), (
            f"{path}: missing. Run `nox -s zoo-docs` to regenerate the model zoo docs."
        )
        assert path.read_text(encoding="utf-8") == content, (
            f"{path}: out of date. Run `nox -s zoo-docs` to regenerate the model "
            f"zoo docs."
        )
    assert entry.slug in (_paths.DOCS_DIR / "index.md").read_text(encoding="utf-8"), (
        f"{_paths.DOCS_DIR / 'index.md'}: does not list '{entry.slug}'. Run "
        f"`nox -s zoo-docs` to regenerate the model zoo docs."
    )


@_register("codeowners", scope=REPO)
def check_codeowners(entry: ModelEntry) -> None:
    """``.github/CODEOWNERS`` names an owner for the model's folder."""
    folder = _paths.codeowners_folder(entry.slug)
    for line in _paths.CODEOWNERS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(folder) and "@" in stripped:
            return
    handle = entry.maintainers[0].github if entry.maintainers else "handle"
    raise AssertionError(
        f"{_paths.CODEOWNERS}: no owner for '{folder}'. Add a line such as "
        f"'{folder} @{handle}' so changes to the model request its maintainer "
        f"for review."
    )


def instantiate(entry: ModelEntry) -> pybamm.BaseModel:
    """Build a fresh instance of a registered model."""
    return check_import(entry)()


def parameter_values_for(
    entry: ModelEntry, model: pybamm.BaseModel
) -> pybamm.ParameterValues:
    """The parameter values the contract exercises a model with.

    The manifest's ``tests.parameter_set`` when it names one, otherwise the
    model's own defaults — which is where a model adds its extra parameters.
    """
    if entry.tests.parameter_set:
        return pybamm.ParameterValues(entry.tests.parameter_set)
    return model.default_parameter_values


def _simulation_for(entry: ModelEntry) -> pybamm.Simulation:
    """A simulation over a fresh instance, as a user would set one up.

    Going through ``Simulation`` rather than driving the parameter, mesh, and
    discretisation steps by hand keeps the ``build`` check exercising the same
    pipeline as the ``solve`` check.
    """
    model = instantiate(entry)
    return pybamm.Simulation(model, parameter_values=parameter_values_for(entry, model))


def missing_dependencies(entry: ModelEntry) -> list[str]:
    """Requirements from the model's extra that are not installed."""
    missing = []
    for item in entry.dependencies.packages:
        requirement = Requirement(item)
        try:
            installed = Version(version(requirement.name))
        except PackageNotFoundError:
            missing.append(item)
            continue
        if not requirement.specifier.contains(installed, prereleases=True):
            missing.append(item)
    return missing
