"""Unit tests for manifest parsing and the registry itself."""

import textwrap
from pathlib import Path

import pytest

import pybamm
import pybamm_model_zoo as zoo
from pybamm_model_zoo._citations import parse_bibtex
from pybamm_model_zoo._registry import Registry
from pybamm_model_zoo.testing import contract

MANIFEST = """
[model]
slug = "{slug}"
name = "{name}"
title = "A title"
summary = "A summary."
class = "pybamm_model_zoo.{slug}:{name}"
tier = "community"
pybamm_requires = ">=26.0"
added = "2026-01-01"
license = "BSD-3-Clause"

[[model.maintainers]]
name = "A. Author"
github = "ahandle"

[model.citation]
key = "Author2026"
"""


def write_model(root: Path, slug: str, name: str, body: str | None = None) -> Path:
    folder = root / slug
    folder.mkdir(parents=True)
    (folder / "model.toml").write_text(
        body if body is not None else MANIFEST.format(slug=slug, name=name)
    )
    return folder


class TestRegistry:
    def test_discovers_the_reference_model(self):
        assert "LinearisedSPM" in zoo.list_models()
        entry = zoo.info("LinearisedSPM")
        assert entry.slug == "linearised_spm"
        assert entry.tier == "core"
        assert entry.maintainers[0].github == "pybamm-team/maintainers"

    def test_defaults_are_applied_for_optional_fields(self, tmp_path):
        write_model(tmp_path, "minimal_model", "MinimalModel")
        entry = Registry([tmp_path])["MinimalModel"]
        assert entry.tier == "community"
        assert entry.tests.solve_time == 3600
        assert entry.tests.key_variables == ("Voltage [V]",)
        assert entry.dependencies.extra is None
        assert not entry.external

    def test_unknown_name_lists_what_is_registered(self, tmp_path):
        write_model(tmp_path, "minimal_model", "MinimalModel")
        with pytest.raises(KeyError, match=r"MinimalModel"):
            Registry([tmp_path])["Nope"]

    def test_by_slug(self, tmp_path):
        write_model(tmp_path, "minimal_model", "MinimalModel")
        registry = Registry([tmp_path])
        assert registry.by_slug("minimal_model").name == "MinimalModel"
        with pytest.raises(KeyError, match=r"minimal_model"):
            registry.by_slug("other")

    def test_invalid_toml_is_reported_with_its_path(self, tmp_path):
        write_model(tmp_path, "broken_model", "Broken", body="[model\nslug =")
        with pytest.raises(zoo.ManifestError, match=r"invalid TOML"):
            Registry([tmp_path])

    def test_missing_model_table_is_reported(self, tmp_path):
        write_model(tmp_path, "broken_model", "Broken", body="[other]\nkey = 1\n")
        with pytest.raises(zoo.ManifestError, match=r"missing a \[model\] table"):
            Registry([tmp_path])

    def test_duplicate_names_are_rejected(self, tmp_path):
        write_model(tmp_path, "one_model", "Same")
        write_model(tmp_path, "two_model", "Same")
        with pytest.raises(zoo.ManifestError, match=r"duplicate model name"):
            Registry([tmp_path])

    def test_duplicate_slugs_are_rejected(self, tmp_path):
        one, two = tmp_path / "one", tmp_path / "two"
        write_model(one, "same_slug", "OneName")
        write_model(two, "same_slug", "OtherName")
        with pytest.raises(zoo.ManifestError, match=r"duplicate model slug"):
            Registry([one, two])

    @pytest.mark.parametrize(
        ("slug", "name"),
        [("minimal_model", "MinimalModel"), ("minimal_model", "Different")],
    )
    def test_external_models_do_not_shadow_in_tree_ones(self, tmp_path, slug, name):
        """Neither key may be shadowed: `by_slug` is what resolves citations."""
        in_tree = tmp_path / "in_tree"
        external = tmp_path / "external"
        write_model(in_tree, "minimal_model", "MinimalModel")
        write_model(external, slug, name)
        with pytest.warns(UserWarning, match=r"ignoring external model"):
            registry = Registry([in_tree], external_paths=[external])
        assert registry.by_slug("minimal_model").path.parent == in_tree
        assert not registry.by_slug("minimal_model").external
        assert name not in registry or registry[name].path.parent == in_tree

    def test_external_entries_are_flagged(self, tmp_path):
        write_model(tmp_path, "minimal_model", "MinimalModel")
        assert Registry([], external_paths=[tmp_path])["MinimalModel"].external


class TestLoad:
    def test_load_returns_the_class(self):
        model_class = zoo.load("LinearisedSPM")
        assert model_class.__name__ == "LinearisedSPM"

    def test_unparseable_class_path(self, tmp_path):
        body = MANIFEST.format(slug="minimal_model", name="MinimalModel").replace(
            'class = "pybamm_model_zoo.minimal_model:MinimalModel"', 'class = "nope"'
        )
        write_model(tmp_path, "minimal_model", "MinimalModel", body=body)
        with pytest.raises(zoo.ManifestError, match=r"module.path:AttributeName"):
            Registry([tmp_path])["MinimalModel"].load()

    def test_missing_module_names_the_extra(self, tmp_path):
        body = MANIFEST.format(
            slug="minimal_model", name="MinimalModel"
        ) + textwrap.dedent(
            """
            [model.dependencies]
            extra = "zoo-minimal-model"
            packages = ["not-a-real-package>=1.0"]
            """
        )
        write_model(tmp_path, "minimal_model", "MinimalModel", body=body)
        entry = Registry([tmp_path])["MinimalModel"]
        with pytest.raises(zoo.ModelUnavailableError, match=r"zoo-minimal-model"):
            entry.load()
        assert contract.missing_dependencies(entry) == ["not-a-real-package>=1.0"]


class TestPybammRequires:
    """The declared range, against installs that can and cannot name a release."""

    def entry(self, tmp_path, requires):
        body = MANIFEST.format(slug="minimal_model", name="MinimalModel").replace(
            'pybamm_requires = ">=26.0"', f'pybamm_requires = "{requires}"'
        )
        write_model(tmp_path, "minimal_model", "MinimalModel", body=body)
        return Registry([tmp_path])["MinimalModel"]

    def test_unsatisfied_range_fails_on_a_real_release(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pybamm, "__version__", "26.8.0.0")
        with pytest.raises(AssertionError, match=r"is not satisfied by"):
            contract.check_pybamm_requires(self.entry(tmp_path, ">=99.0"))

    # setuptools_scm's guess with no tag in reach, and hatch-vcs's own fallback.
    @pytest.mark.parametrize("version", ["0.0.1.dev1+gabc1234", "0.0.0"])
    def test_range_is_not_held_to_an_install_that_names_no_release(
        self, tmp_path, monkeypatch, version
    ):
        assert not contract.names_a_release(version)
        monkeypatch.setattr(pybamm, "__version__", version)
        contract.check_pybamm_requires(self.entry(tmp_path, ">=99.0"))

    def test_invalid_specifier_fails_wherever_it_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pybamm, "__version__", "0.0.1.dev1+gabc1234")
        with pytest.raises(AssertionError, match=r"not a valid specifier"):
            contract.check_pybamm_requires(self.entry(tmp_path, "=>26.0"))

    @pytest.mark.parametrize("version", ["0.1.0", "25.1.0", "26.8.0.1.dev4+gabc1234"])
    def test_a_dev_build_off_a_real_tag_still_names_a_release(self, version):
        assert contract.names_a_release(version)


class TestCitationParsing:
    def test_parses_multiple_entries(self):
        entries = parse_bibtex(
            "@article{A2020, title = {{Nested {braces} here}},}\n"
            "@misc{B2021, note = {x},}\n"
        )
        assert sorted(entries) == ["A2020", "B2021"]
        assert entries["A2020"].startswith("@article{A2020")
        assert entries["A2020"].endswith("}")

    def test_reference_model_citation_resolves(self):
        entry = zoo.info("LinearisedSPM")
        assert entry.citation_key in zoo.read_citations(entry.path)

    def test_register_citation_rejects_an_unknown_key(self):
        with pytest.raises(zoo.ManifestError, match=r"no entry for 'Nope'"):
            zoo.register_citation("linearised_spm", "Nope")
