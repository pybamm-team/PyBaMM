import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[4] / "scripts" / "next_pr_number.py"
spec = importlib.util.spec_from_file_location("next_pr_number", SCRIPT)
next_pr_number = importlib.util.module_from_spec(spec)
spec.loader.exec_module(next_pr_number)


class TestNextPrNumber:
    def test_returns_one_more_than_the_newest_number(self, monkeypatch):
        response = {
            "data": {
                "repository": {
                    "issues": {"nodes": [{"number": 40}]},
                    "pullRequests": {"nodes": [{"number": 42}]},
                    "discussions": {"nodes": [{"number": 41}]},
                }
            }
        }
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(response), stderr=""
        )
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: completed)

        assert next_pr_number.next_pr_number() == 43

    def test_reports_github_cli_failure(self, monkeypatch):
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(
                1, "gh", stderr="authentication required"
            )

        monkeypatch.setattr(subprocess, "run", fail)

        with pytest.raises(SystemExit, match="authentication required"):
            next_pr_number.next_pr_number()
