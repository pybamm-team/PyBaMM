import nox
import os
import sys
from pathlib import Path


# Options to modify nox behaviour
nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True
if sys.platform != "win32":
    nox.options.sessions = ["pre-commit", "pybamm-requires", "unit"]
else:
    nox.options.sessions = ["pre-commit", "unit"]


homedir = os.getenv("HOME")
PYBAMM_ENV = {
    "SUNDIALS_INST": f"{homedir}/.local",
    "LD_LIBRARY_PATH": f"{homedir}/.local/lib",
    "PIP_NO_BINARY": "scikits.odes",
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


@nox.session(name="pybamm-requires")
def run_pybamm_requires(session):
    """Download, compile, and install the build-time requirements for Linux and macOS. Supports --install-dir for custom installation paths and --force to force installation."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        session.install("cmake", silent=False)
        session.run("python", "scripts/install_KLU_Sundials.py", *session.posargs)
        if not os.path.exists("./pybind11"):
            session.run(
                "git",
                "clone",
                "https://github.com/pybind/pybind11.git",
                "pybind11/",
                external=True,
            )
    else:
        session.error("nox -s pybamm-requires is only available on Linux & macOS.")


@nox.session(name="coverage")
def run_coverage(session):
    """Run the coverage tests and generate an XML report."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("coverage", silent=False)
    if sys.platform != "win32":
        if sys.version_info > (3, 12):
            session.install("-e", ".[all,dev,jax]", silent=False)
        else:
            session.run_always(
                sys.executable,
                "-m",
                "pip",
                "cache",
                "remove",
                "scikits.odes",
                external=True,
            )
            session.install("-e", ".[all,dev,jax,odes]", silent=False)
    else:
        if sys.version_info < (3, 9):
            session.install("-e", ".[all,dev]", silent=False)
        else:
            session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("coverage", "run", "run-tests.py", "--nosub")
    session.run("coverage", "combine")
    session.run("coverage", "xml")


@nox.session(name="integration")
def run_integration(session):
    """Run the integration tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        if sys.version_info > (3, 12):
            session.install("-e", ".[all,dev,jax]", silent=False)
        else:
            session.run_always(
                sys.executable,
                "-m",
                "pip",
                "cache",
                "remove",
                "scikits.odes",
                external=True,
            )
            session.install("-e", ".[all,dev,jax,odes]", silent=False)
    else:
        if sys.version_info < (3, 9):
            session.install("-e", ".[all,dev]", silent=False)
        else:
            session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "run-tests.py", "--integration")


@nox.session(name="doctests")
def run_doctests(session):
    """Run the doctests and generate the output(s) in the docs/build/ directory."""
    session.install("-e", ".[all,docs]", silent=False)
    session.run("python", "run-tests.py", "--doctest")


@nox.session(name="unit")
def run_unit(session):
    """Run the unit tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        if sys.version_info > (3, 12):
            session.install("-e", ".[all,dev,jax]", silent=False)
        else:
            session.run_always(
                sys.executable,
                "-m",
                "pip",
                "cache",
                "remove",
                "scikits.odes",
                external=True,
            )
            session.install("-e", ".[all,dev,jax,odes]", silent=False)
    else:
        if sys.version_info < (3, 9):
            session.install("-e", ".[all,dev]", silent=False)
        else:
            session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "run-tests.py", "--unit")


@nox.session(name="examples")
def run_examples(session):
    """Run the examples tests for Jupyter notebooks."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all,dev]", silent=False)
    notebooks_to_test = session.posargs if session.posargs else []
    session.run("pytest", "--nbmake", *notebooks_to_test, external=True)


@nox.session(name="scripts")
def run_scripts(session):
    """Run the scripts tests for Python scripts."""
    set_environment_variables(PYBAMM_ENV, session=session)
    # Temporary fix for Python 3.12 CI. TODO: remove after
    # https://bitbucket.org/pybtex-devs/pybtex/issues/169/replace-pkg_resources-with
    # is fixed
    session.install("setuptools", silent=False)
    session.install("-e", ".[all]", silent=False)
    session.run("python", "run-tests.py", "--scripts")


@nox.session(name="dev")
def set_dev(session):
    """Install PyBaMM in editable mode."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("virtualenv", "cmake")
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)
    python = os.fsdecode(VENV_DIR.joinpath("bin/python"))
    # Temporary fix for Python 3.12 CI. TODO: remove after
    # https://bitbucket.org/pybtex-devs/pybtex/issues/169/replace-pkg_resources-with
    # is fixed
    session.run(python, "-m", "pip", "install", "setuptools", external=True)
    if sys.platform == "linux":
        if sys.version_info > (3, 12):
            session.run(
                python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[all,dev,jax]",
                external=True,
            )
        else:
            session.run_always(
                sys.executable,
                "-m",
                "pip",
                "cache",
                "remove",
                "scikits.odes",
                external=True,
            )
            session.run(
                python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[all,dev,jax,odes]",
                external=True,
            )
    else:
        if sys.version_info < (3, 9):
            session.run(
                python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[all,dev]",
                external=True,
            )
        else:
            session.run(
                python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[all,dev,jax]",
                external=True,
            )


@nox.session(name="tests")
def run_tests(session):
    """Run the unit tests and integration tests sequentially."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        if sys.version_info > (3, 12):
            session.install("-e", ".[all,dev,jax]", silent=False)
        else:
            session.run_always(
                sys.executable,
                "-m",
                "pip",
                "cache",
                "remove",
                "scikits.odes",
                external=True,
            )
            session.install("-e", ".[all,dev,jax,odes]", silent=False)
    else:
        if sys.version_info < (3, 9):
            session.install("-e", ".[all,dev]", silent=False)
        else:
            session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "run-tests.py", "--all")


@nox.session(name="docs")
def build_docs(session):
    """Build the documentation and load it in a browser tab, rebuilding on changes."""
    envbindir = session.bin
    session.install("-e", ".[all,docs]", silent=False)
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


@nox.session(name="pre-commit")
def lint(session):
    """Check all files against the defined pre-commit hooks."""
    session.install("pre-commit", silent=False)
    session.run("pre-commit", "run", "--all-files")


@nox.session(name="quick", reuse_venv=True)
def run_quick(session):
    """Run integration tests, unit tests, and doctests sequentially"""
    run_tests(session)
    run_doctests(session)
