"""The contract suite: every registered model, held to every check.

Contributors write none of this. Adding a model folder with a manifest adds a
column to this matrix automatically, and adding a check to
:data:`pybamm_model_zoo.testing.contract.CHECKS` adds a row.
"""

import pytest

import pybamm_model_zoo as zoo
from pybamm_model_zoo import _docs
from pybamm_model_zoo.testing import contract

# An externally-registered model is held only to the portable rules: it is not
# wired into this package and does not live in this repository.
IN_TREE_SCOPES = (contract.MODEL, contract.PACKAGING, contract.REPO)
EXTERNAL_SCOPES = (contract.MODEL,)


def contract_cases():
    return [
        pytest.param(
            entry,
            check,
            id=f"{entry.slug}-{check.name}",
            marks=pytest.mark.zoo_model(entry.slug),
        )
        for entry in zoo.all_entries()
        for check in contract.checks_in_scope(
            *(EXTERNAL_SCOPES if entry.external else IN_TREE_SCOPES)
        )
    ]


@pytest.mark.parametrize(("entry", "check"), contract_cases())
def test_contract(entry, check):
    if check.name in entry.tests.skip_contract:
        # A reviewed, per-check escape hatch, visible in the manifest diff.
        pytest.skip(f"{entry.slug}: '{check.name}' waived by {entry.manifest_path}")
    if check.needs_model and (missing := contract.missing_dependencies(entry)):
        pytest.skip(
            f"{entry.slug}: extra '{entry.dependencies.extra}' is not installed "
            f"(missing {missing})"
        )
    check.run(entry)


class TestContractItself:
    def test_at_least_one_model_is_registered(self):
        assert zoo.list_models(), (
            "the registry is empty, so the contract suite would vacuously pass"
        )

    def test_every_check_is_well_formed(self):
        assert contract.CHECKS, "the contract is empty"
        for name, check in contract.CHECKS.items():
            assert check.name == name
            assert check.scope in IN_TREE_SCOPES, f"{name}: unknown scope"
            assert check.run.__doc__, (
                f"{name}: needs a docstring saying what it asserts"
            )


class TestGeneratedFiles:
    """The index page and the absence of leftovers, which no per-model check sees."""

    def test_docs_and_badges_are_current(self):
        files = _docs.all_files(zoo.all_entries())
        outdated = [
            path
            for path, content in files.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        assert not outdated + _docs.stale(files), (
            "out of date, run `nox -s zoo-docs`: "
            f"{sorted(str(path) for path in outdated + _docs.stale(files))}"
        )
