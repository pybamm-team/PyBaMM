"""The contract suite: every registered model, held to every check.

Contributors write none of this. Adding a model folder with a manifest adds a
column to this matrix automatically, and adding a check to
:data:`pybamm_model_zoo.testing.contract.CHECKS` adds a row.
"""

import re
from pathlib import Path

import pytest
import yaml

import pybamm_model_zoo as zoo
from pybamm_model_zoo import _docs, _paths
from pybamm_model_zoo._registry import MANIFEST_NAME, ModelEntry
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
    if entry.error is not None and check.name != "manifest":
        # Nothing else has inputs to check, so 'manifest' carries the one failure.
        pytest.skip(f"{entry.slug}: manifest did not parse")
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


class TestDependencyAgreement:
    """A manifest and the extra behind it, held to each other in both directions."""

    def check(self, tmp_path, packages, extra_items):
        entry = ModelEntry(
            slug="a_model",
            name="AModel",
            path=tmp_path,
            raw={
                "model": {
                    "dependencies": {"extra": "zoo-a-model", "packages": packages}
                }
            },
        )
        contract._check_requirements_agree(
            entry, tmp_path / "pyproject.toml", extra_items
        )

    def test_matching_requirements_pass(self, tmp_path):
        self.check(tmp_path, ["scikit-fem>=12.0.2"], ["scikit-fem>=12.0.2"])

    def test_a_package_the_extra_omits_is_caught(self, tmp_path):
        with pytest.raises(AssertionError, match=r"is missing \['scikit-fem'\]"):
            self.check(tmp_path, ["scikit-fem>=12.0.2"], [])

    def test_a_package_the_manifest_omits_is_caught(self, tmp_path):
        with pytest.raises(AssertionError, match=r"does not declare \['scikit-fem'\]"):
            self.check(tmp_path, [], ["scikit-fem>=12.0.2"])

    def test_a_disagreeing_constraint_is_caught(self, tmp_path):
        with pytest.raises(AssertionError, match=r"declared differently"):
            self.check(tmp_path, ["scikit-fem>=13"], ["scikit-fem>=12.0.2"])

    def test_names_are_compared_canonically(self, tmp_path):
        self.check(tmp_path, ["Scikit_FEM>=12.0.2"], ["scikit-fem>=12.0.2"])


class TestMissingDependencies:
    """What counts as "not installed", which gates the import/build/solve checks."""

    def entry(self, packages):
        return ModelEntry(
            slug="a_model",
            name="AModel",
            path=Path("a_model"),
            raw={"model": {"dependencies": {"packages": packages}}},
        )

    def test_an_absent_package_is_missing(self):
        assert contract.missing_dependencies(
            self.entry(["definitely-not-installed-xyz"])
        ) == ["definitely-not-installed-xyz"]

    def test_an_installed_package_is_not_missing(self):
        assert contract.missing_dependencies(self.entry(["packaging>=23.0"])) == []

    def test_a_requirement_whose_marker_is_false_is_not_missing(self):
        """Otherwise a platform-specific dependency skips the checks everywhere else."""
        assert (
            contract.missing_dependencies(
                self.entry(['definitely-not-installed-xyz; python_version < "3.0"'])
            )
            == []
        )

    def test_a_requirement_whose_marker_is_true_is_still_checked(self):
        assert contract.missing_dependencies(
            self.entry(['definitely-not-installed-xyz; python_version >= "3.0"'])
        ) == ['definitely-not-installed-xyz; python_version >= "3.0"']


class TestDocsHookWatchesWhatItRegenerates:
    """The pre-commit hook's `files:` pattern, against the generator's real I/O.

    The pattern went stale once already by naming `scripts/generate.py` while the
    rendering lived in `_docs`, so it is derived from the files rather than
    listed by hand.
    """

    HOOK = "model-zoo-docs"

    @pytest.fixture
    def pattern(self):
        config = yaml.safe_load(
            (_paths.REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
        )
        hooks = [
            hook
            for repo in config["repos"]
            for hook in repo["hooks"]
            if hook["id"] == self.HOOK
        ]
        assert len(hooks) == 1, f"expected exactly one '{self.HOOK}' hook"
        return re.compile(hooks[0]["files"])

    def watched(self, pattern, path):
        return bool(pattern.match(path.relative_to(_paths.REPO_ROOT).as_posix()))

    def test_it_watches_everything_the_generator_writes(self, pattern):
        for path in _docs.all_files(zoo.all_entries()):
            assert self.watched(pattern, path), f"{path} is regenerated but unwatched"

    def test_it_watches_everything_the_generator_reads(self, pattern):
        sources = [
            _paths.ZOO_ROOT / "scripts" / "generate.py",
            _paths.STATUS_FILE,
            *_paths.PACKAGE_ROOT.rglob("*.py"),
            *_paths.PACKAGE_ROOT.glob(f"*/{MANIFEST_NAME}"),
            *_paths.PACKAGE_ROOT.glob("*/README.md"),
        ]
        for path in sources:
            assert self.watched(pattern, path), (
                f"{path} feeds the docs but is unwatched"
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
