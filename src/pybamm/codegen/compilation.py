import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid

import casadi

from pybamm import logger

_CACHE: dict[str, casadi.Function] = {}

# Only remove build artifacts older than this to avoid racing with another
# process's in-flight compile.
_STALE_TMP_AGE_S = 3600

# Per-attempt temp filenames have the form ``<stem>.<pid>.<32-hex-uuid>.c``
# or ``...<ext>.tmp``.
_PER_ATTEMPT_TOKEN = re.compile(r"\.\d+\.[0-9a-f]{32}(?:\.|$)")

_TMP_FILE_PREFIX = "pybamm_"

_swept_dirs: set[str] = set()


def _default_cache_dir() -> str:
    d = os.environ.get("PYBAMM_CASADI_AOT_CACHE")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    d = os.path.join(tempfile.gettempdir(), f"{_TMP_FILE_PREFIX}casadi_aot")
    os.makedirs(d, exist_ok=True)
    return d


def _shared_ext() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def aot_compile(fn: casadi.Function, **kwargs) -> casadi.Function:
    """Ahead-of-time compile a casadi ``Function`` to a shared library and
    return a ``casadi.external`` wrapper.

    Results are cached in-process (by a hash of the serialised form) and on
    disk under ``$PYBAMM_CASADI_AOT_CACHE`` (default
    ``$TMPDIR/pybamm_casadi_aot``). ``casadi.external`` inputs are returned
    unchanged. Any failure is logged and the original ``fn`` returned.

    Parameters
    ----------
    fn : casadi.Function
        The function to compile.
    **kwargs
        ``cache_dir``, ``compiler`` and ``flags`` overrides forwarded to
        :func:`_aot_compile`.

    Returns
    -------
    casadi.Function
    """
    try:
        return _aot_compile(fn, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to compile function {fn.name()} with error: {e}")
        return fn


def _aot_compile(
    fn: casadi.Function,
    *,
    cache_dir: str | None = None,
    compiler: str | None = None,
    flags: tuple[str, ...] | None = None,
) -> casadi.Function:
    # Already an External: don't recompile. ``fn.is_a("External")`` is
    # unreliable, so check the class name directly.
    if fn.class_name() == "External":
        return fn

    if compiler is None:
        compiler = "gcc"

    if flags is None:
        flags = ("-O3", "-march=native", "-fPIC")

    key = hashlib.sha1(fn.serialize().encode()).hexdigest()[:16]
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    cdir = cache_dir or _default_cache_dir()
    _maybe_sweep_stale(cdir)

    stem = f"pybamm_{fn.name()}_{key}"
    ext = _shared_ext()
    sofile = os.path.join(cdir, stem + ext)

    if not os.path.exists(sofile):
        gen = casadi.CodeGenerator(stem, {"with_header": False})
        gen.add(fn)
        c_source = gen.dump()

        # Per-attempt temp paths so concurrent compiles of the same function
        # can't clobber each other, and so an interrupted build can be
        # detected and cleaned up later.
        suffix = f".{os.getpid()}.{uuid.uuid4().hex}"
        tmp_cfile = os.path.join(cdir, stem + suffix + ".c")
        tmp_sofile = os.path.join(cdir, stem + suffix + ext + ".tmp")
        try:
            with open(tmp_cfile, "w") as f:
                f.write(c_source)
            subprocess.run(
                [compiler, *flags, "-shared", tmp_cfile, "-o", tmp_sofile],
                check=True,
            )
            os.replace(tmp_sofile, sofile)
            if os.environ.get("PYBAMM_CASADI_AOT_KEEP_C"):
                os.replace(tmp_cfile, os.path.join(cdir, stem + ".c"))
        finally:
            for p in (tmp_cfile, tmp_sofile):
                try:
                    os.remove(p)
                except OSError:
                    pass

    ext_fn = casadi.external(fn.name(), sofile)
    _CACHE[key] = ext_fn
    return ext_fn


def _maybe_sweep_stale(cdir: str) -> None:
    # Remove leaked per-attempt artifacts and orphan .c files once per
    # process. Only touches files matching our naming, and only if older
    # than ``_STALE_TMP_AGE_S``.
    if cdir in _swept_dirs:
        return
    _swept_dirs.add(cdir)

    try:
        entries = os.listdir(cdir)
    except OSError:
        return

    cutoff = time.time() - _STALE_TMP_AGE_S
    ext = _shared_ext()
    have_so = {n for n in entries if n.endswith(ext) and n.startswith(_TMP_FILE_PREFIX)}

    for name in entries:
        if not name.startswith(_TMP_FILE_PREFIX):
            continue
        path = os.path.join(cdir, name)
        try:
            if os.path.getmtime(path) > cutoff:
                continue

            is_per_attempt = bool(_PER_ATTEMPT_TOKEN.search(name))
            if is_per_attempt and (name.endswith(".tmp") or name.endswith(".c")):
                os.remove(path)
                continue

            if name.endswith(".c") and not is_per_attempt:
                stem = name[: -len(".c")]
                if (stem + ext) not in have_so:
                    os.remove(path)
        except OSError:
            pass
