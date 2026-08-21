"""Render the template and hold the result to the whole portable contract.

This is what makes "follow the template and CI is green on day one" a tested
claim rather than a hope: if a contract check and the template ever disagree,
this test fails instead of the next contributor's first pull request. The checks
are read from the contract registry, so a new check cannot quietly skip the
template.
"""

import re
import shutil
import subprocess  # nosec B404 - runs the repo's own ruff over a rendered template
import sys

import pytest

import pybamm_model_zoo as zoo
from pybamm_model_zoo import _paths, _template
from pybamm_model_zoo.testing import contract

SLUG = "template_smoke_model"
NAME = "TemplateSmokeModel"
# Everything but the REPO scope: a freshly rendered model has no docs page or
# CODEOWNERS line until it is committed.
TEMPLATE_SCOPES = (contract.MODEL, contract.PACKAGING)


@pytest.fixture
def rendered(tmp_path):
    """A rendered template, importable as ``pybamm_model_zoo.<slug>``."""
    values = _template.tokens(
        slug=SLUG,
        name=NAME,
        author="A. Author",
        github="ahandle",
        year=2026,
        added="2026-01-01",
    )
    _template.render(tmp_path / SLUG, values)
    # Extending the package's search path is what lets the manifest's in-tree
    # `pybamm_model_zoo.<slug>` class path resolve from a temporary directory.
    zoo.__path__.append(str(tmp_path))
    try:
        yield zoo.refresh([tmp_path]).by_slug(SLUG)
    finally:
        zoo.__path__.remove(str(tmp_path))
        if hasattr(zoo, SLUG):
            delattr(zoo, SLUG)
        _forget_module(f"{zoo.__name__}.{SLUG}")
        zoo.refresh()


def _forget_module(prefix):
    for name in [
        name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")
    ]:
        del sys.modules[name]


class TestTemplate:
    def test_renders_every_file(self, rendered):
        for name in ("model.toml", "README.md", "CITATION.bib", "__init__.py"):
            assert (rendered.path / name).is_file()
        assert (rendered.path / "examples" / f"run_{SLUG}.py").is_file()
        assert (rendered.path / "tests" / f"test_{SLUG}.py").is_file()

    def test_leaves_no_unsubstituted_placeholders(self, rendered):
        for path in sorted(rendered.path.rglob("*")):
            if path.is_file():
                assert not _template.PLACEHOLDER_PATTERN.search(
                    path.read_text(encoding="utf-8")
                ), f"{path}: unsubstituted template placeholder"

    @pytest.mark.parametrize(
        "check",
        contract.checks_in_scope(*TEMPLATE_SCOPES),
        ids=lambda check: check.name,
    )
    def test_contract(self, rendered, check):
        check.run(rendered)

    def test_rejects_a_bad_slug(self):
        with pytest.raises(zoo.ZooError, match=r"lower_snake_case"):
            _template.tokens(slug="MyModel", name="MyModel", author="A", github="a")

    def test_codeowners_line_names_the_contributor(self):
        line = _template.codeowners_line(SLUG, "@ahandle")
        assert line.endswith(" @ahandle")
        assert line.startswith(_paths.codeowners_folder(SLUG))


class TestTokenValidation:
    """The scaffold refuses inputs it would render into invalid Python."""

    def tokens(self, **overrides):
        return _template.tokens(
            **{
                "slug": SLUG,
                "name": NAME,
                "author": "A. Author",
                "github": "ahandle",
                **overrides,
            }
        )

    @pytest.mark.parametrize("keyword_name", ["class", "import", "lambda", "None"])
    def test_a_python_keyword_is_rejected(self, keyword_name):
        """`class` is identifier-shaped, so only a keyword check catches it."""
        with pytest.raises(zoo.ZooError, match=r"keyword"):
            self.tokens(slug=keyword_name.lower(), name=keyword_name)

    @pytest.mark.parametrize("soft_keyword", ["match", "case", "type"])
    def test_a_soft_keyword_is_still_a_usable_name(self, soft_keyword):
        assert self.tokens(slug=soft_keyword, name=soft_keyword)["slug"] == soft_keyword

    def test_the_default_floor_pins_the_minor_release(self):
        """A bare major would let a model claim releases predating its own APIs."""
        requires = _template.default_pybamm_requires()
        assert re.fullmatch(r">=\d+\.\d+", requires), requires

    def test_an_explicit_specifier_wins_over_the_default(self):
        assert self.tokens(pybamm_requires=">=26.4")["pybamm_requires"] == ">=26.4"


class TestRenderedStyle:
    """A rendered template must also pass the repository's style job.

    The template files are not themselves linted (they carry placeholders and a
    `.in` suffix), so without this the skeleton could drift out of Ruff's
    formatting and a contributor's first commit would be reformatted under them.
    """

    @pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
    @pytest.mark.parametrize(
        "command", [("check", "--no-cache"), ("format", "--check", "--no-cache")]
    )
    def test_rendered_python_is_style_clean(self, rendered, command):
        result = subprocess.run(  # nosec B603 B607 - literal argv, no external input
            [
                "ruff",
                *command,
                "--config",
                str(_paths.REPO_ROOT / "pyproject.toml"),
                ".",
            ],
            cwd=rendered.path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
