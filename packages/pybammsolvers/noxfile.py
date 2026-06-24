import nox
import sys
from pathlib import Path


nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
if sys.platform != "win32":
    nox.options.sessions = ["idaklu-requires", "unit"]
else:
    nox.options.sessions = ["unit"]

homedir = Path(__file__).parent.resolve()
PYBAMM_ENV = {
    "LD_LIBRARY_PATH": f"{homedir}/.idaklu/lib",
    "DYLD_LIBRARY_PATH": f"{homedir}/.idaklu/lib",
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
}
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


# Build deps from pyproject.toml + cmake/ninja (scikit-build-core provides them isolated but needed for --no-build-isolation)
_pyproject = nox.project.load_toml("pyproject.toml")
BUILD_DEPS = (*_pyproject["build-system"]["requires"], "cmake>=3.13", "ninja")


def editable_install(session, *extras, no_deps=False):
    """Install pybammsolvers in editable mode without build isolation.

    Pre-installing BUILD_DEPS into the session venv and passing
    --no-build-isolation prevents scikit-build-core's editable.rebuild
    shim from baking dangled paths (cmake/ninja in pip's ephemeral build
    env) into CMakeCache.txt. `uv sync` users get the same behaviour
    automatically via [tool.uv].no-build-isolation-package.
    """
    session.install(*BUILD_DEPS, silent=False)
    target = "." if not extras else f".[{','.join(extras)}]"
    install_args = ["-e", target, "--no-build-isolation"]
    if no_deps:
        install_args.append("--no-deps")
    session.install(*install_args, silent=False)


@nox.session(name="idaklu-requires")
def run_pybamm_requires(session):
    """Download, compile, and install the build-time requirements for Linux and macOS.
    Supports --install-dir for custom installation paths and --force to force installation."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py", *session.posargs)
    else:
        session.error("nox -s idaklu-requires is only available on Linux & macOS.")


@nox.session(name="unit")
def run_unit(session):
    """Run the full test suite."""
    set_environment_variables(PYBAMM_ENV, session=session)
    editable_install(session, "dev")
    session.run("pytest", "tests", "-m", "unit", *session.posargs)


@nox.session(name="coverage")
def run_coverage(session):
    """Run tests with coverage reporting."""
    set_environment_variables(PYBAMM_ENV, session=session)
    editable_install(session, "dev")
    session.install("pytest-cov", silent=False)
    session.run(
        "pytest",
        "tests",
        "--cov=pybammsolvers",
        "--cov-report=html",
        "--cov-report=term-missing",
        *session.posargs,
    )


@nox.session(name="integration")
def run_integration(session):
    """Run integration tests"""
    set_environment_variables(PYBAMM_ENV, session=session)

    # Build and install pybammsolvers first
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py")

    editable_install(session, "dev")
    session.install("pybamm", silent=False)

    # Reinstall local pybammsolvers since pybamm pulls it from PyPI
    editable_install(session, no_deps=True)

    # Run integration tests
    session.run("pytest", "tests", "-m", "integration", *session.posargs)


@nox.session(name="benchmarks")
def run_benchmarks(session):
    """Run PyBaMM performance benchmarks comparing vanilla PyBaMM vs pybammsolvers.

    This session:
    1. Runs benchmarks with vanilla PyBaMM (baseline)
    2. Installs local pybammsolvers
    3. Re-runs benchmarks with pybammsolvers
    4. Compares results and reports regressions
    """
    set_environment_variables(PYBAMM_ENV, session=session)

    # Build and install pybammsolvers first (for compilation)
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py")

    editable_install(session, "dev")

    # Install PyBaMM
    session.install("pybamm", silent=False)

    # Reinstall local pybammsolvers since pybamm pulls it from PyPI
    editable_install(session, no_deps=True)

    # Run the benchmark orchestrator script
    session.run("python", "tests/pybamm_benchmarks/run_benchmarks.py", *session.posargs)


@nox.session(name="dev-rebuild", venv_backend="none")
def run_dev_rebuild(session):
    """Rebuild the C++ extension in-place against the active venv.

    Escape hatch when scikit-build-core's editable.rebuild auto-rebuild
    is bypassed. Requires build deps in the active venv
    (`uv sync --extra dev`).
    """
    set_environment_variables(PYBAMM_ENV, session=session)
    session.run(
        "uv",
        "pip",
        "install",
        "-e",
        ".",
        "--no-deps",
        "--no-build-isolation",
        external=True,
    )
