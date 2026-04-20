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

# Cache of bundle-hash -> list of ``casadi.external`` wrappers, one per
# non-External input to that bundle. A single-Function call is a bundle of
# size one; no separate code path.
_CACHE: dict[str, list[casadi.Function]] = {}

# Only remove build artifacts older than this to avoid racing with another
# process's in-flight compile.
_STALE_TMP_AGE_S = 3600

# Per-attempt temp filenames have the form ``<stem>.<pid>.<32-hex-uuid>.c``
# or ``...<ext>.tmp``.
_PER_ATTEMPT_TOKEN = re.compile(r"\.\d+\.[0-9a-f]{32}(?:\.|$)")

_TMP_FILE_PREFIX = "pybamm_"

# ``int NAME(const casadi_real** arg, ...);`` at the top level of the
# generated C marks an External sub-Function named ``NAME``. Decls for names
# defined in the same TU are fine; decls for anything else mean the caller
# wrapped an inner Function as an External before feeding it to a composite.
_EXTERN_DECL = re.compile(
    r"^\s*int\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*const\s+casadi_real\s*\*\*",
    re.MULTILINE,
)

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


def aot_compile(fn_or_fns, **kwargs):
    """Ahead-of-time compile one or more casadi ``Function`` objects to a
    single shared library and return ``casadi.external`` wrappers.

    Accepts either a single ``casadi.Function`` (returns a Function) or a
    list/tuple of Functions (returns a list, one per input, in order). In
    either case everything is lowered in one ``CodeGenerator`` / ``gcc``
    invocation -- a single fn is a bundle of size one.

    Intended for the *outermost* Functions a solver hands off (e.g.
    ``rhs_algebraic``, ``jac_times_cjmass``, ``rootfn``, output-variable
    evaluators). Intermediate Functions should stay as MX/SX so
    ``casadi.CodeGenerator`` can inline them into one translation unit.
    Wrapping inner Functions as Externals forces cross-dylib dispatch and
    produces unresolvable ``extern`` declarations.

    Results are cached in-process (by a hash of the serialised forms) and on
    disk under ``$PYBAMM_CASADI_AOT_CACHE`` (default
    ``$TMPDIR/pybamm_casadi_aot``). Inputs already of class ``External`` are
    returned unchanged. On any failure, the original Function(s) are returned
    and a warning is logged.

    Parameters
    ----------
    fn_or_fns : casadi.Function or list of casadi.Function
    **kwargs
        ``cache_dir``, ``compiler`` and ``flags`` overrides.
    """
    is_single = isinstance(fn_or_fns, casadi.Function)
    fns = [fn_or_fns] if is_single else list(fn_or_fns)
    try:
        out = _aot_compile(fns, **kwargs)
    except Exception as e:
        names = ", ".join(fn.name() for fn in fns)
        logger.warning(f"Failed to compile [{names}] with error: {e}")
        out = list(fns)
    return out[0] if is_single else out


def _aot_compile(
    fns: list[casadi.Function],
    *,
    cache_dir: str | None = None,
    compiler: str | None = None,
    flags: tuple[str, ...] | None = None,
) -> list[casadi.Function]:
    # Pass-through Externals; compile the rest together in one TU.
    result: list[casadi.Function] = list(fns)
    indices_to_compile = [
        i for i, fn in enumerate(fns) if fn.class_name() != "External"
    ]
    if not indices_to_compile:
        return result

    # Cache key: ordered hash of each fn's name + serialized form.
    hasher = hashlib.sha1(usedforsecurity=False)
    for idx in indices_to_compile:
        fn = fns[idx]
        hasher.update(fn.name().encode())
        hasher.update(b"\0")
        hasher.update(fn.serialize().encode())
        hasher.update(b"\0")
    key = hasher.hexdigest()[:16]

    cached = _CACHE.get(key)
    if cached is not None:
        for idx, ext_fn in zip(indices_to_compile, cached, strict=True):
            result[idx] = ext_fn
        return result

    if compiler is None:
        compiler = "gcc"
    if flags is None:
        flags = ("-O3", "-march=native", "-fPIC")

    cdir = cache_dir or _default_cache_dir()
    _maybe_sweep_stale(cdir)

    # Single-fn bundles get named after the fn for readability; multi-fn
    # bundles are hash-only since the member list isn't knowable from the
    # filename anyway.
    fns_to_compile = [fns[idx] for idx in indices_to_compile]
    label = fns_to_compile[0].name() if len(fns_to_compile) == 1 else "bundle"
    stem = f"{_TMP_FILE_PREFIX}{label}_{key}"
    ext = _shared_ext()
    sofile = os.path.join(cdir, stem + ext)

    if not os.path.exists(sofile):
        gen = casadi.CodeGenerator(stem, {"with_header": False})
        for fn in fns_to_compile:
            gen.add(fn)
        c_source = gen.dump()

        bundled = {fn.name() for fn in fns_to_compile}
        externs = set(_EXTERN_DECL.findall(c_source)) - bundled
        if externs:
            raise RuntimeError(
                f"References to External sub-Function(s) {sorted(externs)} "
                "cannot be linked. aot_compile should only be called on "
                "top-level Functions; keep intermediate Functions as MX/SX."
            )

        # Per-attempt temp paths so concurrent compiles of the same bundle
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

    ext_fns: list[casadi.Function] = []
    for idx, fn in zip(indices_to_compile, fns_to_compile, strict=True):
        ext_fn = casadi.external(fn.name(), sofile)
        result[idx] = ext_fn
        ext_fns.append(ext_fn)
    _CACHE[key] = ext_fns
    return result


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
