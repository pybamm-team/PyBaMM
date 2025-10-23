import nox
import os
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
    # Expression evaluators (...EXPR_CASADI cannot be fully disabled at this time)
    "PYBAMM_IDAKLU_EXPR_CASADI": os.getenv("PYBAMM_IDAKLU_EXPR_CASADI", "ON"),
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
    session.install("setuptools", silent=False)
    session.install(".[dev]", silent=False)
    session.run("pytest", "tests", "-m", "unit", *session.posargs)


@nox.session(name="coverage")
def run_coverage(session):
    """Run tests with coverage reporting."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install(".[dev]", silent=False)
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

    session.install("setuptools", silent=False)
    session.install(".[dev]", silent=False)

    # Install PyBaMM
    session.install("pybamm", silent=False)

    # Force PyBaMM to use our local pybammsolvers
    session.run("uv", "pip", "uninstall", "pybammsolvers", silent=True)
    session.install("-e", ".", "--no-deps", silent=False)

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

    session.install("setuptools", silent=False)
    session.install(".[dev]", silent=False)

    # Install PyBaMM
    session.install("pybamm", silent=False)

    # Run the benchmark orchestrator script
    session.run("python", "tests/pybamm_benchmarks/run_benchmarks.py", *session.posargs)


@nox.session(name="pybamm-tests")
def run_pybamm_tests(session):
    """Run PyBaMM's full test suite with local pybammsolvers.

    1. Clones PyBaMM repository (if not already present)
    2. Updates to latest develop version (unless --no-update is specified)
    3. Installs PyBaMM with all dependencies
    4. Replaces bundled pybammsolvers with local version
    5. Runs PyBaMM's unit and integration tests

    Usage:
        nox -s pybamm-tests                    # Clone/update PyBaMM and run all tests
        nox -s pybamm-tests -- --unit-only     # Run only unit tests
        nox -s pybamm-tests -- --integration-only  # Run only integration tests
        nox -s pybamm-tests -- --no-update     # Skip git pull (use current version)
        nox -s pybamm-tests -- --pybamm-dir ./custom/path  # Use existing PyBaMM clone
        nox -s pybamm-tests -- --branch develop  # Use specific branch/tag
    """
    set_environment_variables(PYBAMM_ENV, session=session)

    # Parse session arguments
    pybamm_dir = None
    unit_only = False
    integration_only = False
    no_update = False
    branch = "develop"
    pytest_args = []

    i = 0
    while i < len(session.posargs):
        arg = session.posargs[i]
        if arg == "--pybamm-dir" and i + 1 < len(session.posargs):
            pybamm_dir = Path(session.posargs[i + 1]).resolve()
            i += 2
        elif arg == "--branch" and i + 1 < len(session.posargs):
            branch = session.posargs[i + 1]
            i += 2
        elif arg == "--unit-only":
            unit_only = True
            i += 1
        elif arg == "--integration-only":
            integration_only = True
            i += 1
        elif arg == "--no-update":
            no_update = True
            i += 1
        else:
            pytest_args.append(arg)
            i += 1

    # Set default PyBaMM directory
    if pybamm_dir is None:
        pybamm_dir = Path("./PyBaMM").resolve()

    session.log(f"Using PyBaMM directory: {pybamm_dir}")

    # Clone PyBaMM if it doesn't exist
    if not pybamm_dir.exists():
        session.log(f"Cloning PyBaMM repository (branch: {branch})...")
        session.run(
            "git",
            "clone",
            "--branch",
            branch,
            "https://github.com/pybamm-team/PyBaMM.git",
            str(pybamm_dir),
            external=True,
        )
    else:
        session.log(f"PyBaMM directory already exists at {pybamm_dir}")

        # Update PyBaMM if requested (default behavior)
        if not no_update:
            session.log("Updating PyBaMM to latest version...")
            session.cd(str(pybamm_dir))
            try:
                # Fetch latest changes
                session.run("git", "fetch", "--all", "--tags", external=True)
                # Check if we're on the requested branch, switch if needed
                session.run("git", "checkout", branch, external=True)
                # Pull latest changes
                session.run("git", "pull", "--ff-only", external=True)
                session.cd(str(Path(__file__).parent))
            except Exception as e:
                session.warn(f"Could not update PyBaMM: {e}")
                session.warn("Continuing with current version...")
                session.cd(str(Path(__file__).parent))
        else:
            session.log("Skipping PyBaMM update (--no-update specified)")

    # Install PyBaMM with all dependencies
    session.log("Installing PyBaMM with all dependencies...")
    session.cd(str(pybamm_dir))
    session.install("-e", ".[all,dev,jax]", silent=False)

    # Go back to pybammsolvers root
    session.cd(str(Path(__file__).parent))

    # Uninstall bundled pybammsolvers
    session.log("Uninstalling bundled pybammsolvers...")
    session.run("uv", "pip", "uninstall", "pybammsolvers", silent=False)

    # Build and install local pybammsolvers
    session.log("Building and installing local pybammsolvers...")
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py")
    else:
        session.warn("Skipping install_KLU_Sundials.py on Windows")

    # Install local pybammsolvers
    session.install(".", silent=False)

    # Run PyBaMM tests
    session.cd(str(pybamm_dir))

    if unit_only:
        session.log("Running PyBaMM unit tests...")
        session.run("pytest", "tests/unit", *pytest_args)
    elif integration_only:
        session.log("Running PyBaMM integration tests...")
        session.run("pytest", "tests/integration", *pytest_args)
    else:
        session.log("Running PyBaMM unit tests...")
        session.run("pytest", "tests/unit", *pytest_args)
        session.log("Running PyBaMM integration tests...")
        session.run("pytest", "tests/integration", *pytest_args)
