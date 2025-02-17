import nox
import os
import sys
from pathlib import Path


# Options to modify nox behaviour
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["pre-commit", "unit"]

homedir = os.getenv("HOME")
PYBAMM_ENV = {
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


@nox.session(name="coverage")
def run_coverage(session):
    """Run the coverage tests and generate an XML report."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("coverage", silent=False)
    # Using plugin here since coverage runs unit tests on linux with latest python version.
    if "CI" in os.environ:
        session.install("pytest-github-actions-annotate-failures")
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("pytest", "--cov=pybamm", "--cov-report=xml", "tests/unit")


@nox.session(name="integration")
def run_integration(session):
    """Run the integration tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if (
        "CI" in os.environ
        and sys.version_info[:2] == (3, 12)
        and sys.platform == "linux"
    ):
        session.install("pytest-github-actions-annotate-failures")
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "-m", "pytest", "-m", "integration")


@nox.session(name="doctests")
def run_doctests(session):
    """Run the doctests and generate the output(s) in the docs/build/ directory."""
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev,docs]", silent=False)
    session.run(
        "python",
        "-m",
        "pytest",
        "--doctest-plus",
        "src",
    )


@nox.session(name="unit")
def run_unit(session):
    """Run the unit tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "-m", "pytest", "-m", "unit")


@nox.session(name="examples")
def run_examples(session):
    """Run the examples tests for Jupyter notebooks."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all,dev,jax]", silent=False)
    notebooks_to_test = session.posargs if session.posargs else []
    session.run(
        "pytest", "--nbmake", *notebooks_to_test, "docs/source/examples/", external=True
    )


@nox.session(name="scripts")
def run_scripts(session):
    """Run the scripts tests for Python scripts."""
    set_environment_variables(PYBAMM_ENV, session=session)
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "-m", "pytest", "-m", "scripts")


@nox.session(name="dev")
def set_dev(session):
    """Install PyBaMM in editable mode."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("virtualenv", "cmake")
    session.run("virtualenv", os.fsdecode(VENV_DIR), silent=True)
    python = os.fsdecode(VENV_DIR.joinpath("bin/python"))
    components = ["all", "dev", "jax"]
    args = []
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.run(python, "-m", "pip", "install", "setuptools", external=True)
    session.run(
        python,
        "-m",
        "pip",
        "install",
        "-e",
        ".[{}]".format(",".join(components)),
        *args,
        external=True,
    )


@nox.session(name="tests")
def run_tests(session):
    """Run the unit tests and integration tests sequentially."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run(
        "python",
        "-m",
        "pytest",
        *(session.posargs if session.posargs else ["-m", "unit or integration"]),
    )


@nox.session(name="docs")
def build_docs(session):
    """Build the documentation and load it in a browser tab, rebuilding on changes."""
    envbindir = session.bin
    # Fix for Python 3.12 CI. This can be removed after pybtex is replaced.
    session.install("setuptools", silent=False)
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
