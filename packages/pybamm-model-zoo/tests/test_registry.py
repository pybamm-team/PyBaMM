"""Unit tests for manifest parsing and the registry itself."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import pybamm_model_zoo as zoo
from pybamm_model_zoo import _dependencies, _paths, _registry
from pybamm_model_zoo._citations import parse_bibtex
from pybamm_model_zoo._registry import ModelEntry, Registry, read_manifest
from pybamm_model_zoo.testing import contract

MANIFEST = """
[model]
slug = "{slug}"
name = "{name}"
title = "A title"
summary = "A summary."
class = "pybamm_model_zoo.{slug}:{name}"
tier = "community"
added = "2026-01-01"
license = "BSD-3-Clause"

[[model.maintainers]]
name = "A. Author"
github = "ahandle"

[model.citation]
key = "Author2026"
"""


def declare_extra(monkeypatch, distribution: str, extra: str) -> None:
    """Make ``distribution`` declare ``extra`` as a package that is not installed."""
    real_requires = _dependencies.requires

    def requires(name):
        if name != distribution:
            return real_requires(name)
        return [f'not-a-real-package>=1.0; extra == "{extra}"']

    monkeypatch.setattr(_dependencies, "requires", requires)


def write_model(root: Path, slug: str, name: str, body: str | None = None) -> Path:
    folder = root / slug
    folder.mkdir(parents=True)
    (folder / "model.toml").write_text(
        body if body is not None else MANIFEST.format(slug=slug, name=name)
    )
    return folder


class TestPackage:
    def test_version_is_the_pyproject_version(self):
        project = read_manifest(_paths.ZOO_PYPROJECT)["project"]
        assert zoo.__version__ == project["version"]


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
        assert entry.extra == "zoo-minimal-model"
        assert entry.distribution == "pybamm-model-zoo"
        assert not entry.external

    @pytest.mark.parametrize(
        ("declared", "gates", "community"),
        [("core", True, False), ("community", False, True), ("Core", True, True)],
    )
    def test_an_unrecognised_tier_belongs_to_every_tier(
        self, tmp_path, declared, gates, community
    ):
        """A tier typo must fail the gate loudly, not drop quietly out of it."""
        write_model(
            tmp_path,
            "minimal_model",
            "MinimalModel",
            body=MANIFEST.format(slug="minimal_model", name="MinimalModel").replace(
                'tier = "community"', f'tier = "{declared}"'
            ),
        )
        entry = Registry([tmp_path])["MinimalModel"]
        assert entry.in_tier("core") is gates
        assert entry.in_tier("community") is community

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

    @pytest.mark.parametrize(
        ("body", "reported"),
        [
            ("[model\nslug =", r"invalid TOML"),
            ("[other]\nkey = 1\n", r"missing a \[model\] table"),
            ('[model]\nname = "Broken"\n', r"\[model\].slug must be a non-empty"),
            ('[model]\nslug = "broken_model"\n', r"\[model\].name must be a non-empty"),
        ],
    )
    def test_a_manifest_too_broken_to_key_is_kept_and_reported(
        self, tmp_path, body, reported
    ):
        """It is recorded on the entry, not raised: one bad file fails one model."""
        write_model(tmp_path, "broken_model", "Broken", body=body)
        entry = Registry([tmp_path]).by_slug("broken_model")
        assert entry.error is not None
        assert re.search(reported, entry.error), entry.error
        with pytest.raises(AssertionError, match=reported):
            contract.check_manifest(entry)

    def test_one_broken_manifest_does_not_take_down_the_registry(self, tmp_path):
        write_model(tmp_path, "broken_model", "Broken", body="[model\nslug =")
        write_model(tmp_path, "good_model", "GoodModel")
        registry = Registry([tmp_path])
        assert registry["GoodModel"].error is None
        assert sorted(registry) == ["GoodModel", "broken_model"]

    def test_a_broken_manifest_is_never_pruned_out_of_a_tier(self, tmp_path):
        """It declares no trustworthy tier, so every tier has to keep it."""
        write_model(tmp_path, "broken_model", "Broken", body="[model\nslug =")
        entry = Registry([tmp_path]).by_slug("broken_model")
        assert entry.in_tier("core") and entry.in_tier("community")

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
        entry = Registry([], external_paths=[tmp_path])["MinimalModel"]
        assert entry.external
        assert entry.distribution is None

    def test_an_external_collection_records_its_distribution(
        self, tmp_path, monkeypatch
    ):
        """It is what declares the collection's extras, so `load` can name it."""
        package = tmp_path / "lab_models"
        write_model(package, "minimal_model", "MinimalModel")
        (package / "__init__.py").write_text("")
        monkeypatch.syspath_prepend(str(tmp_path))
        entry_point = SimpleNamespace(
            name="lab", value="lab_models", dist=SimpleNamespace(name="lab-models")
        )
        monkeypatch.setattr(_registry, "_iter_entry_points", lambda: [entry_point])
        entry = Registry([])["MinimalModel"]
        assert entry.external
        assert entry.distribution == "lab-models"


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

    def test_a_module_that_does_not_import_is_reported(self, tmp_path):
        write_model(tmp_path, "minimal_model", "MinimalModel")
        with pytest.raises(zoo.ModelUnavailableError, match=r"could not be imported"):
            Registry([tmp_path])["MinimalModel"].load()

    def test_a_missing_extra_fails_before_the_import(self, tmp_path, monkeypatch):
        """The import would succeed: PyBaMM imports scikit-fem only once meshing."""
        declare_extra(monkeypatch, "pybamm-model-zoo", "zoo-minimal-model")
        write_model(tmp_path, "minimal_model", "MinimalModel")
        with pytest.raises(
            zoo.ModelUnavailableError,
            match=re.escape(
                "'MinimalModel' is missing not-a-real-package>=1.0. Install its "
                'extra with `pip install "pybamm-model-zoo[zoo-minimal-model]"`.'
            ),
        ):
            Registry([tmp_path])["MinimalModel"].load()

    def test_an_external_model_names_its_own_distribution(self, tmp_path, monkeypatch):
        declare_extra(monkeypatch, "lab-models", "zoo-minimal-model")
        entry = ModelEntry(
            slug="minimal_model",
            name="MinimalModel",
            path=tmp_path,
            external=True,
            distribution="lab-models",
        )
        with pytest.raises(zoo.ModelUnavailableError) as info:
            entry.require()
        assert 'pip install "lab-models[zoo-minimal-model]"' in str(info.value)

    def test_a_directly_imported_model_fails_before_building(self, monkeypatch):
        """`require` in `__init__` covers the import path that bypasses `load`."""
        model_class = zoo.load("LinearisedSPM")
        declare_extra(monkeypatch, "pybamm-model-zoo", "zoo-linearised-spm")
        with pytest.raises(zoo.ModelUnavailableError, match=r"zoo-linearised-spm"):
            model_class()


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
