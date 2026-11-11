import runpy
from pathlib import Path


class TestDocsConfig:
    def test_stable_notebook_links_use_namespaced_release_tag(self, monkeypatch):
        docs_directory = Path(__file__).parents[4] / "docs"
        monkeypatch.chdir(docs_directory)
        monkeypatch.setenv("READTHEDOCS_VERSION", "stable")
        monkeypatch.delenv("READTHEDOCS_VERSION_TYPE", raising=False)

        config = runpy.run_path("conf.py")

        version = config["version"]
        expected_url = f"https://github.com/pybamm-team/PyBaMM/blob/pybamm-v{version}"
        assert config["github_download_url"] == expected_url
        assert config["google_colab_url"] == expected_url.replace(
            "github.com", "githubtocolab.com"
        )
