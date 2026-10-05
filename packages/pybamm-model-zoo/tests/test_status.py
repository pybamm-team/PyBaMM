"""The weekly compatibility status: folding matrix results into badges."""

import json

from pybamm_model_zoo import _docs


def write_results(directory, records):
    directory.mkdir(exist_ok=True)
    for record in records:
        path = directory / f"{record['model']}--{record['version']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
    return directory


class TestCollectResults:
    def test_folds_one_file_per_cell(self, tmp_path):
        results = write_results(
            tmp_path / "results",
            [
                {"model": "a_model", "version": "26.7.0", "result": "pass"},
                {"model": "a_model", "version": "main", "result": "fail"},
            ],
        )
        status = _docs.collect_results(results)
        assert status["models"]["a_model"]["results"] == {
            "26.7.0": "pass",
            "main": "fail",
        }
        assert status["models"]["a_model"]["last_pass"] == "26.7.0"

    def test_an_expected_cell_that_reported_nothing_is_recorded(self, tmp_path):
        results = write_results(
            tmp_path / "results",
            [{"model": "a_model", "version": "26.7.0", "result": "pass"}],
        )
        expected = [
            {"model": "a_model", "version": "26.7.0"},
            {"model": "a_model", "version": "26.8.0"},
            {"model": "b_model", "version": "main"},
        ]
        status = _docs.collect_results(results, expected=expected)
        assert status["models"]["a_model"]["results"]["26.8.0"] == _docs.MISSING
        assert status["models"]["b_model"]["results"] == {"main": _docs.MISSING}

    def test_a_reported_cell_is_not_overwritten_as_missing(self, tmp_path):
        results = write_results(
            tmp_path / "results",
            [{"model": "a_model", "version": "main", "result": "fail"}],
        )
        status = _docs.collect_results(
            results, expected=[{"model": "a_model", "version": "main"}]
        )
        assert status["models"]["a_model"]["results"] == {"main": "fail"}


class TestBadge:
    def test_all_passing_is_green(self):
        record = {"results": {"26.7.0": "pass"}, "last_pass": "26.7.0"}
        assert _docs.badge(record)["color"] == _docs.BADGE_COLORS["pass"]

    def test_a_missing_cell_does_not_read_as_passing(self):
        record = {"results": {"26.7.0": "pass", "main": _docs.MISSING}}
        assert _docs.badge(record)["color"] == _docs.BADGE_COLORS["missing"]
        assert "main" in _docs.badge(record)["message"]

    def test_a_failure_outranks_a_missing_cell(self):
        record = {"results": {"26.7.0": "fail", "main": _docs.MISSING}}
        assert _docs.badge(record)["color"] == _docs.BADGE_COLORS["fail"]
        assert "failing on 26.7.0" in _docs.badge(record)["message"]

    def test_no_results_at_all_is_untested(self):
        assert _docs.badge({})["message"] == "untested"


class TestStamp:
    RESULTS = {"a_model": {"results": {"main": "pass"}, "last_pass": None}}

    def test_an_unchanged_result_keeps_its_timestamp(self):
        previous = {"generated": "2026-01-01T00:00:00Z", "models": self.RESULTS}
        stamped = _docs.stamp(
            {"models": self.RESULTS}, previous, "2026-02-02T00:00:00Z"
        )
        assert stamped["generated"] == "2026-01-01T00:00:00Z"

    def test_a_changed_result_is_restamped(self):
        previous = {"generated": "2026-01-01T00:00:00Z", "models": self.RESULTS}
        changed = {"a_model": {"results": {"main": "fail"}, "last_pass": None}}
        stamped = _docs.stamp({"models": changed}, previous, "2026-02-02T00:00:00Z")
        assert stamped["generated"] == "2026-02-02T00:00:00Z"

    def test_a_first_run_is_stamped(self):
        stamped = _docs.stamp({"models": {}}, {"models": {}}, "2026-02-02T00:00:00Z")
        assert stamped["generated"] == "2026-02-02T00:00:00Z"
