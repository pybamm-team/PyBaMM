import os
import platform
import sys
from pathlib import Path

import nox

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.needs_version = ">= 2025.10.14"

homedir = os.getenv("HOME")
PYBAMM_ENV = {
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
    "PYBAMM_TEST_FORBID_LEGACY_OPTION_DEFAULTS": "1",
}
# Headless Linux renders VTK through OSMesa (the CI workflows install libosmesa6)
if sys.platform.startswith("linux") and not (
    os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
):
    PYBAMM_ENV["VTK_DEFAULT_OPENGL_WINDOW"] = os.getenv(
        "VTK_DEFAULT_OPENGL_WINDOW", "vtkOSOpenGLRenderWindow"
    )
VENV_DIR = Path("./venv").resolve()


def set_environment_variables(env_dict, session):
    """
    Sets environment variables for a nox Session object.

    Parameters
    -----------
        session : nox.Session
            The session to set the environment variables for.
        env_dict : dict
            A dictionary of environment variable names and values.

    """
    for key, value in env_dict.items():
        session.env[key] = value


def is_macos_intel():
    """Check whether the current interpreter is running on macOS Intel (x86_64)."""
    return sys.platform == "darwin" and platform.machine() in ("x86_64", "i386")


def install_locked(session, *, extras=None, groups=None, zoo_extras=None):
    """Install every workspace member and its ``uv.lock`` dependencies.

    ``uv sync --frozen`` installs the locked environment, building
    ``pybammsolvers`` from source. When ``PYBAMM_SOLVER_WHEELS`` names a wheel
    or a directory of wheels (the CI matrix), the sync skips ``pybammsolvers``
    and this interpreter's prebuilt wheel is installed in its place, since
    Windows has no from-source build.
    """
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    wheels = os.getenv("PYBAMM_SOLVER_WHEELS")

    cmd = ["uv", "sync", "--frozen"]
    for extra in [*(extras or []), *(zoo_extras or [])]:
        cmd.extend(["--extra", extra])
    for group in groups or []:
        cmd.extend(["--group", group])
    if wheels:
        cmd.extend(["--no-install-package", "pybammsolvers"])
    session.run(*cmd, env=env, external=True)

    if not wheels:
        return
    wheels_path = Path(wheels)
    if wheels_path.is_dir():
        # The trailing dash keeps free-threaded "cpXYt-" wheels out of the match
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        matches = sorted(wheels_path.glob(f"*-{tag}-*.whl"))
        if not matches:
            session.error(
                f"PYBAMM_SOLVER_WHEELS={wheels} contains no pybammsolvers "
                f"wheel for {tag}"
            )
        wheel = matches[0].as_posix()
    else:
        wheel = wheels_path.as_posix()
    python_bin = os.path.join(
        session.bin, "python.exe" if sys.platform == "win32" else "python"
    )
    # The sync already installed the solver's locked dependencies
    session.run(
        "uv",
        "pip",
        "install",
        "--python",
        python_bin,
        "--no-deps",
        wheel,
        env=env,
        external=True,
    )


@nox.session(name="coverage", default=False)
def run_coverage(session):
    """Run the coverage tests and generate an XML report."""
    set_environment_variables(PYBAMM_ENV, session=session)
    # sys.monitoring core (Python 3.12+) slashes coverage tracing overhead.
    session.env["COVERAGE_CORE"] = "sysmon"
    install_locked(session, extras=["all", "jax"], groups=["dev"])
    # Using plugin here since coverage runs unit tests on linux with latest python version.
    if "CI" in os.environ:
        session.install("pytest-github-actions-annotate-failures")
    session.run(
        "pytest", "--cov=pybamm", "--cov-report=xml", "packages/pybamm/tests/unit"
    )


@nox.session(name="integration", default=False)
def run_integration(session):
    """Run the integration tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    extras = ["all", "jax"]
    # pydiffsol has no working build on macOS Intel CI runners.
    if not is_macos_intel():
        extras.append("pydiffsol")
    install_locked(session, extras=extras, groups=["dev"])
    if (
        "CI" in os.environ
        and sys.version_info[:2] >= (3, 12)
        and sys.platform == "linux"
    ):
        session.install("pytest-github-actions-annotate-failures")
    session.run("python", "-m", "pytest", "-m", "integration", "packages/pybamm/tests")


@nox.session(name="doctests", default=False)
def run_doctests(session):
    """Run the doctests and generate the output(s) in the docs/build/ directory."""
    install_locked(session, extras=["all"], groups=["dev", "docs"])
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
    session.run(
        "python",
        "-m",
        "pytest",
        "--doctest-plus",
        "packages/pybamm/src",
    )


@nox.session(name="unit", default=True)
def run_unit(session):
    """Run the unit tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, extras=["all", "jax"], groups=["dev"])
    session.run("python", "-m", "pytest", "-m", "unit", "packages/pybamm/tests")


@nox.session(name="memory", default=False)
def run_memory(session):
    """Run memory leak tests using pytest-memray (Linux/macOS only)."""
    if sys.platform == "win32":
        session.skip("memray is not supported on Windows")
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, groups=["dev"])
    session.run(
        "python",
        "-m",
        "pytest",
        "packages/pybamm/tests/memory/",
        "-v",
        "-o",
        "addopts=",
    )


@nox.session(name="benchmark-time", default=False)
def run_benchmark_time(session):
    """Run timing benchmark tests locally.

    CI does not use this session: Bencher runs the same pytest command inside the
    benchmark image on bare metal hardware (see tests/benchmarks/Dockerfile).
    """
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, groups=["dev"])
    session.run(
        "python",
        "-m",
        "pytest",
        "packages/pybamm/tests/benchmarks/",
        "-m",
        "time_bench",
        "-v",
        "-o",
        "addopts=",
        "--benchmark-group-by",
        "func",
        "--benchmark-disable-gc",
        *session.posargs,
    )


@nox.session(name="benchmark-memory", default=False)
def run_benchmark_memory(session):
    """Run memory benchmarks with memray (Linux/macOS only)."""
    if sys.platform == "win32":
        session.skip("memray is not supported on Windows")
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, groups=["dev"])
    session.run(
        "python",
        "-m",
        "pytest",
        "packages/pybamm/tests/benchmarks/",
        "-m",
        "memory_bench",
        "-v",
        "-o",
        "addopts=",
        "--memray",
        # Without this memray sees pymalloc only as whole 1 MiB arenas, so
        # results jump by 1 MiB depending on heap state left by earlier tests.
        "--trace-python-allocators",
        "--benchmark-disable",
    )


@nox.session(name="examples", default=False)
def run_examples(session):
    """Run the examples tests for Jupyter notebooks."""
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, extras=["all", "jax"], groups=["dev"])
    notebooks_to_test = session.posargs if session.posargs else []
    session.run(
        "pytest", "--nbmake", *notebooks_to_test, "docs/source/examples/", external=True
    )


@nox.session(name="scripts", default=False)
def run_scripts(session):
    """Run the scripts tests for Python scripts."""
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, extras=["all", "jax"], groups=["dev"])
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
    session.run("python", "-m", "pytest", "-m", "scripts", "packages/pybamm/tests")


@nox.session(name="dev", default=False)
def set_dev(session):
    """Install PyBaMM in editable mode."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.run(
        "uv",
        "sync",
        "--frozen",
        "--extra",
        "all",
        "--extra",
        "jax",
        "--group",
        "dev",
        env={"UV_PROJECT_ENVIRONMENT": os.fsdecode(VENV_DIR)},
        external=True,
    )


@nox.session(name="tests", default=False)
def run_tests(session):
    """Run the unit tests and integration tests sequentially."""
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, extras=["all", "jax"], groups=["dev"])
    session.run(
        "python",
        "-m",
        "pytest",
        *(
            session.posargs
            if session.posargs
            else ["-m", "unit or integration", "packages/pybamm/tests"]
        ),
    )


@nox.session(name="docs", default=False)
def build_docs(session):
    """Build the documentation and load it in a browser tab, rebuilding on changes."""
    envbindir = session.bin
    install_locked(session, extras=["all"], groups=["docs"])
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
    session.chdir("docs")
    # Local development
    if session.interactive:
        session.run(
            "sphinx-autobuild",
            "-j",
            "auto",
            "--open-browser",
            "-qT",
            ".",
            f"{envbindir}/../tmp/html",
        )
    # Runs in CI only, treating warnings as errors
    # Run in single-threaded mode, see
    # https://github.com/pydata/pydata-sphinx-theme/issues/1643
    else:
        session.run(
            "sphinx-build",
            "-b",
            "html",
            "-W",
            "--keep-going",
            ".",
            f"{envbindir}/../tmp/html",
        )


ZOO_TESTS = "packages/pybamm-model-zoo"


def install_zoo(session):
    """Install pybamm plus the zoo and every model's declared dependencies."""
    set_environment_variables(PYBAMM_ENV, session=session)
    install_locked(session, extras=["all"], groups=["dev"], zoo_extras=["zoo-all"])


def zoo_pytest(session, marker, *args, allow_empty=False):
    """Run the zoo suite, selecting by marker."""
    session.run(
        "python",
        "-m",
        "pytest",
        "-m",
        marker,
        *args,
        ZOO_TESTS,
        *session.posargs,
        # 5 is "collected nothing", which is the honest state of the community
        # tier until someone contributes to it, not a failure.
        success_codes=[0, 5] if allow_empty else [0],
    )


@nox.session(name="zoo", default=False)
def run_zoo(session):
    """Run the whole model zoo suite: contract, model tests, and examples."""
    install_zoo(session)
    zoo_pytest(session, "zoo")


@nox.session(name="zoo-gating", default=False)
def run_zoo_gating(session):
    """Run what is in PyBaMM's merge gate: `core`-tier models and the zoo itself.

    `--zoo-tier` keeps every other tier out of collection entirely, so a model
    the gate does not cover cannot break the gate by failing to import.
    """
    install_zoo(session)
    zoo_pytest(session, "zoo and gating", "--zoo-tier=core")


@nox.session(name="zoo-community", default=False)
def run_zoo_community(session):
    """Run the advisory half: everything the merge gate does not already cover."""
    install_zoo(session)
    zoo_pytest(session, "zoo and not gating", allow_empty=True)


@nox.session(name="zoo-examples", default=False)
def run_zoo_examples(session):
    """Run every model zoo example script."""
    install_zoo(session)
    zoo_pytest(session, "zoo_examples")


# No install: the generator reads manifests with tomllib and never imports pybamm,
# so building an environment for it would cost minutes to do milliseconds of work.
@nox.session(name="zoo-docs", default=False, venv_backend="none")
def run_zoo_docs(session):
    """Regenerate the model zoo docs pages and badges from the manifests."""
    session.run("python", f"{ZOO_TESTS}/scripts/generate.py", *session.posargs)


@nox.session(name="zoo-new", default=False)
def run_zoo_new(session):
    """Create a new model zoo entry from the template."""
    # Only the zoo itself and pybamm's version metadata are needed, so this skips
    # the extras and the dev group that install_zoo pulls in.
    install_locked(session)
    session.run("python", f"{ZOO_TESTS}/scripts/new_model.py", *session.posargs)


@nox.session(name="pre-commit", default=True)
def lint(session):
    """Check all files against the defined pre-commit hooks."""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="quick", reuse_venv=True, default=False)
def run_quick(session):
    """Run integration tests, unit tests, and doctests sequentially"""
    run_tests(session)
    run_doctests(session)


if __name__ == "__main__":
    nox.main()
