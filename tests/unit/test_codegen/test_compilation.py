#
# Tests for ahead-of-time (AOT) compilation of casadi functions
#
import logging
import os
import shutil
import time

import casadi
import numpy as np
import pytest

from pybamm.codegen import compilation as compilation_module
from pybamm.codegen.compilation import (
    _CACHE,
    _PER_ATTEMPT_TOKEN,
    _STALE_TMP_AGE_S,
    _default_cache_dir,
    _maybe_sweep_stale,
    _shared_ext,
    _swept_dirs,
    aot_compile,
)


def _has_compiler():
    return shutil.which("gcc") is not None or shutil.which("cc") is not None


pytestmark = pytest.mark.skipif(
    not _has_compiler(), reason="No C compiler available for AOT compilation"
)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "aot_cache"
    d.mkdir()
    return str(d)


@pytest.fixture(autouse=True)
def _clear_in_memory_cache():
    cache_snapshot = dict(_CACHE)
    swept_snapshot = set(_swept_dirs)
    _CACHE.clear()
    _swept_dirs.clear()
    try:
        yield
    finally:
        _CACHE.clear()
        _CACHE.update(cache_snapshot)
        _swept_dirs.clear()
        _swept_dirs.update(swept_snapshot)


def _make_simple_fn(name="test_aot_simple"):
    x = casadi.MX.sym("x", 3)
    y = casadi.MX.sym("y")
    expr = casadi.sin(x) + y * x
    return casadi.Function(name, [x, y], [expr])


class TestAotCompile:
    def test_compiles_to_external(self, cache_dir):
        f = _make_simple_fn("test_aot_compiles_to_external")
        assert f.class_name() != "External"

        g = aot_compile(f, cache_dir=cache_dir)

        assert g.class_name() == "External"
        produced = [p for p in os.listdir(cache_dir) if p.endswith(_shared_ext())]
        assert len(produced) == 1
        assert f.name() in produced[0]

    def test_outputs_match_original(self, cache_dir):
        f = _make_simple_fn("test_aot_outputs_match")
        g = aot_compile(f, cache_dir=cache_dir)

        rng = np.random.default_rng(0)
        for _ in range(5):
            xv = rng.standard_normal(3)
            yv = float(rng.standard_normal())
            np.testing.assert_allclose(
                np.array(g(xv, yv)).flatten(),
                np.array(f(xv, yv)).flatten(),
                rtol=1e-12,
                atol=1e-12,
            )

    def test_in_memory_cache_hit(self, cache_dir):
        f = _make_simple_fn("test_aot_inmem_cache")
        g1 = aot_compile(f, cache_dir=cache_dir)
        g2 = aot_compile(f, cache_dir=cache_dir)
        assert g1 is g2

    def test_on_disk_cache_skips_recompile(self, cache_dir):
        f = _make_simple_fn("test_aot_disk_cache")
        _ = aot_compile(f, cache_dir=cache_dir)
        sofile = next(
            os.path.join(cache_dir, p)
            for p in os.listdir(cache_dir)
            if p.endswith(_shared_ext())
        )
        mtime_before = os.path.getmtime(sofile)

        _CACHE.clear()

        called = []
        original_run = compilation_module.subprocess.run

        def fake_run(*args, **kwargs):  # pragma: no cover - fail path
            called.append(args)
            return original_run(*args, **kwargs)

        compilation_module.subprocess.run = fake_run
        try:
            g2 = aot_compile(f, cache_dir=cache_dir)
        finally:
            compilation_module.subprocess.run = original_run

        assert called == [], "gcc was invoked despite an existing on-disk .dylib"
        assert g2.class_name() == "External"
        assert os.path.getmtime(sofile) == mtime_before
        np.testing.assert_allclose(
            np.array(g2(np.array([0.1, 0.2, 0.3]), 1.5)).flatten(),
            np.array(f(np.array([0.1, 0.2, 0.3]), 1.5)).flatten(),
        )

    def test_external_input_passes_through(self, cache_dir):
        f = _make_simple_fn("test_aot_external_passthrough")
        g = aot_compile(f, cache_dir=cache_dir)
        assert g.class_name() == "External"

        n_dylibs_before = sum(
            1 for p in os.listdir(cache_dir) if p.endswith(_shared_ext())
        )

        called = []
        original_run = compilation_module.subprocess.run

        def fake_run(*args, **kwargs):  # pragma: no cover - fail path
            called.append(args)
            return original_run(*args, **kwargs)

        compilation_module.subprocess.run = fake_run
        try:
            g2 = aot_compile(g, cache_dir=cache_dir)
        finally:
            compilation_module.subprocess.run = original_run

        assert g2 is g
        assert called == []
        n_dylibs_after = sum(
            1 for p in os.listdir(cache_dir) if p.endswith(_shared_ext())
        )
        assert n_dylibs_after == n_dylibs_before

    def test_compiler_failure_returns_original(self, cache_dir, caplog):
        f = _make_simple_fn("test_aot_compiler_failure")
        with caplog.at_level(logging.WARNING, logger="pybamm.logger"):
            g = aot_compile(f, cache_dir=cache_dir, compiler="nonexistent_compiler_x")
        assert g is f
        assert not any(p.endswith(_shared_ext()) for p in os.listdir(cache_dir))
        assert len(_CACHE) == 0
        assert any(
            "Failed to compile function" in r.getMessage() for r in caplog.records
        )

    def test_atomic_install_no_partial_dylib_on_failure(self, cache_dir):
        f = _make_simple_fn("test_aot_atomic_install")

        original_run = compilation_module.subprocess.run

        def failing_run(args, *a, **kw):
            output_path = args[args.index("-o") + 1]
            with open(output_path, "wb") as fh:
                fh.write(b"not a valid shared library")
            raise compilation_module.subprocess.CalledProcessError(1, args)

        compilation_module.subprocess.run = failing_run
        try:
            g = aot_compile(f, cache_dir=cache_dir)
        finally:
            compilation_module.subprocess.run = original_run

        assert g is f
        assert not any(p.endswith(_shared_ext()) for p in os.listdir(cache_dir))
        assert not any(p.endswith(".tmp") for p in os.listdir(cache_dir))

    def test_default_cache_dir_env_override(self, tmp_path, monkeypatch):
        target = tmp_path / "envcache" / "deep"
        monkeypatch.setenv("PYBAMM_CASADI_AOT_CACHE", str(target))
        d = _default_cache_dir()
        assert d == str(target)
        assert os.path.isdir(d)

    def test_different_functions_get_distinct_cache_entries(self, cache_dir):
        f1 = _make_simple_fn("test_aot_distinct_a")
        f2 = _make_simple_fn("test_aot_distinct_b")
        g1 = aot_compile(f1, cache_dir=cache_dir)
        g2 = aot_compile(f2, cache_dir=cache_dir)
        assert g1 is not g2
        assert len(_CACHE) == 2


class TestAotCleanup:
    def test_c_source_is_deleted_after_successful_compile(self, cache_dir):
        f = _make_simple_fn("test_aot_c_deleted")
        aot_compile(f, cache_dir=cache_dir)

        files = os.listdir(cache_dir)
        assert not any(p.endswith(".c") for p in files)
        assert not any(p.endswith(".tmp") for p in files)
        assert sum(1 for p in files if p.endswith(_shared_ext())) == 1

    def test_c_source_is_kept_when_env_var_is_set(self, cache_dir, monkeypatch):
        monkeypatch.setenv("PYBAMM_CASADI_AOT_KEEP_C", "1")
        f = _make_simple_fn("test_aot_c_kept")
        aot_compile(f, cache_dir=cache_dir)

        files = os.listdir(cache_dir)
        c_files = [p for p in files if p.endswith(".c")]
        assert len(c_files) == 1
        assert not _PER_ATTEMPT_TOKEN.search(c_files[0])
        assert sum(1 for p in files if p.endswith(_shared_ext())) == 1

    def test_per_attempt_paths_used_for_concurrency_safety(self, cache_dir):
        f = _make_simple_fn("test_aot_per_attempt_paths")

        captured = {}
        original_run = compilation_module.subprocess.run

        def spying_run(args, *a, **kw):
            captured["args"] = list(args)
            return original_run(args, *a, **kw)

        compilation_module.subprocess.run = spying_run
        try:
            aot_compile(f, cache_dir=cache_dir)
        finally:
            compilation_module.subprocess.run = original_run

        gcc_args = captured["args"]
        cfile = next(a for a in gcc_args if a.endswith(".c"))
        tmp_sofile = gcc_args[gcc_args.index("-o") + 1]

        assert _PER_ATTEMPT_TOKEN.search(os.path.basename(cfile))
        assert _PER_ATTEMPT_TOKEN.search(os.path.basename(tmp_sofile))
        assert tmp_sofile != os.path.join(
            cache_dir, "pybamm_test_aot_per_attempt_paths_*"
        )

    def test_sweep_removes_stale_tmp_files(self, cache_dir):
        old_tmp = os.path.join(
            cache_dir,
            f"pybamm_oldfn_0123456789abcdef.999.{'a' * 32}{_shared_ext()}.tmp",
        )
        with open(old_tmp, "wb") as fh:
            fh.write(b"junk from a killed build")
        old = time.time() - _STALE_TMP_AGE_S - 60
        os.utime(old_tmp, (old, old))

        _maybe_sweep_stale(cache_dir)

        assert not os.path.exists(old_tmp)

    def test_sweep_removes_stale_per_attempt_c_files(self, cache_dir):
        old_c = os.path.join(
            cache_dir,
            f"pybamm_oldfn_0123456789abcdef.999.{'b' * 32}.c",
        )
        with open(old_c, "w") as fh:
            fh.write("/* junk */")
        old = time.time() - _STALE_TMP_AGE_S - 60
        os.utime(old_c, (old, old))

        _maybe_sweep_stale(cache_dir)

        assert not os.path.exists(old_c)

    def test_sweep_removes_canonical_orphan_c_files(self, cache_dir):
        orphan_c = os.path.join(cache_dir, "pybamm_oldfn_0123456789abcdef.c")
        with open(orphan_c, "w") as fh:
            fh.write("/* orphan */")
        old = time.time() - _STALE_TMP_AGE_S - 60
        os.utime(orphan_c, (old, old))

        _maybe_sweep_stale(cache_dir)

        assert not os.path.exists(orphan_c)

    def test_sweep_keeps_canonical_c_when_dylib_exists(self, cache_dir):
        stem = "pybamm_oldfn_0123456789abcdef"
        c_path = os.path.join(cache_dir, stem + ".c")
        so_path = os.path.join(cache_dir, stem + _shared_ext())
        for p in (c_path, so_path):
            with open(p, "wb") as fh:
                fh.write(b"x")
        old = time.time() - _STALE_TMP_AGE_S - 60
        os.utime(c_path, (old, old))
        os.utime(so_path, (old, old))

        _maybe_sweep_stale(cache_dir)

        assert os.path.exists(c_path)
        assert os.path.exists(so_path)

    def test_sweep_keeps_recent_files(self, cache_dir):
        recent_tmp = os.path.join(
            cache_dir,
            f"pybamm_inflight_0123456789abcdef.111.{'c' * 32}{_shared_ext()}.tmp",
        )
        recent_c = os.path.join(
            cache_dir,
            f"pybamm_inflight_0123456789abcdef.111.{'d' * 32}.c",
        )
        for p in (recent_tmp, recent_c):
            with open(p, "wb") as fh:
                fh.write(b"in-flight build")

        _maybe_sweep_stale(cache_dir)

        assert os.path.exists(recent_tmp)
        assert os.path.exists(recent_c)

    def test_sweep_ignores_unrelated_files(self, cache_dir):
        unrelated = [
            os.path.join(cache_dir, "important_user_file.c"),
            os.path.join(cache_dir, "important_user_file.tmp"),
            os.path.join(cache_dir, "notes.txt"),
            os.path.join(cache_dir, "pybamm_no_extension"),
        ]
        for p in unrelated:
            with open(p, "wb") as fh:
                fh.write(b"keep me")
            old = time.time() - _STALE_TMP_AGE_S - 60
            os.utime(p, (old, old))

        _maybe_sweep_stale(cache_dir)

        for p in unrelated:
            assert os.path.exists(p)

    def test_sweep_runs_at_most_once_per_cdir(self, cache_dir, monkeypatch):
        calls = []
        original_listdir = os.listdir

        def spying_listdir(path):
            if path == cache_dir:
                calls.append(path)
            return original_listdir(path)

        monkeypatch.setattr(os, "listdir", spying_listdir)

        _maybe_sweep_stale(cache_dir)
        _maybe_sweep_stale(cache_dir)
        _maybe_sweep_stale(cache_dir)

        assert len(calls) == 1

    def test_aot_compile_triggers_sweep(self, cache_dir):
        old_tmp = os.path.join(
            cache_dir,
            f"pybamm_legacy_0123456789abcdef.999.{'e' * 32}{_shared_ext()}.tmp",
        )
        with open(old_tmp, "wb") as fh:
            fh.write(b"junk")
        old = time.time() - _STALE_TMP_AGE_S - 60
        os.utime(old_tmp, (old, old))

        aot_compile(_make_simple_fn("test_aot_sweep_trigger"), cache_dir=cache_dir)

        assert not os.path.exists(old_tmp)
