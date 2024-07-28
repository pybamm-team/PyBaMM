import nox
import os
import sys
import warnings
import platform
from pathlib import Path


# Options to modify nox behaviour
nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True
if sys.platform != "win32":
    nox.options.sessions = ["pre-commit", "pybamm-requires", "unit"]
else:
    nox.options.sessions = ["pre-commit", "unit"]


def set_iree_state():
    """
    Check if IREE is enabled and set the environment variable accordingly.

    Returns
    -------
    str
        "ON" if IREE is enabled, "OFF" otherwise.

    """
    state = "ON" if os.getenv("PYBAMM_IDAKLU_EXPR_IREE", "OFF") == "ON" else "OFF"
    if state == "ON":
        if sys.platform == "win32":
            warnings.warn(
                (
                    "IREE is not enabled on Windows yet. "
                    "Setting PYBAMM_IDAKLU_EXPR_IREE=OFF."
                ),
                stacklevel=2,
            )
            return "OFF"
        if sys.platform == "darwin":
            # iree-compiler is currently only available as a wheel on macOS 13 (or
            # higher) and Python version 3.11
            mac_ver = int(platform.mac_ver()[0].split(".")[0])
            if (not sys.version_info[:2] == (3, 11)) or mac_ver < 13:
                warnings.warn(
                    (
                        "IREE is only supported on MacOS 13 (or higher) and Python"
                        "version 3.11. Setting PYBAMM_IDAKLU_EXPR_IREE=OFF."
                    ),
                    stacklevel=2,
                )
                return "OFF"
    return state


homedir = os.getenv("HOME")
PYBAMM_ENV = {
    "LD_LIBRARY_PATH": f"{homedir}/.local/lib",
    "PYTHONIOENCODING": "utf-8",
    "MPLBACKEND": "Agg",
    # Expression evaluators (...EXPR_CASADI cannot be fully disabled at this time)
    "PYBAMM_IDAKLU_EXPR_CASADI": os.getenv("PYBAMM_IDAKLU_EXPR_CASADI", "ON"),
    "PYBAMM_IDAKLU_EXPR_IREE": set_iree_state(),
    "IREE_INDEX_URL": os.getenv(
        "IREE_INDEX_URL", "https://iree.dev/pip-release-links.html"
    ),
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
                "--depth",
                "1",
                "--branch",
                "v2.12.0",
                "https://github.com/pybind/pybind11.git",
                "pybind11/",
                "-c",
                "advice.detachedHead=false",
                external=True,
            )
        if PYBAMM_ENV.get("PYBAMM_IDAKLU_EXPR_IREE") == "ON" and not os.path.exists(
            "./iree"
        ):
            session.run(
                "git",
                "clone",
                "--depth=1",
                "--recurse-submodules",
                "--shallow-submodules",
                "--branch=candidate-20240507.886",
                "https://github.com/openxla/iree",
                "iree/",
                external=True,
            )
            with session.chdir("iree"):
                session.run(
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    external=True,
                )
    else:
        session.error("nox -s pybamm-requires is only available on Linux & macOS.")


@nox.session(name="coverage")
def run_coverage(session):
    """Run the coverage tests and generate an XML report."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("coverage", silent=False)
    # Using plugin here since coverage runs unit tests on linux with latest python version.
    if "CI" in os.environ:
        session.install("pytest-github-actions-annotate-failures")
    session.install("-e", ".[all,dev,jax]", silent=False)
    if PYBAMM_ENV.get("PYBAMM_IDAKLU_EXPR_IREE") == "ON":
        # See comments in 'dev' session
        session.install(
            "-e",
            ".[iree]",
            "--find-links",
            PYBAMM_ENV.get("IREE_INDEX_URL"),
            silent=False,
        )
    session.run("pytest", "--cov=pybamm", "--cov-report=xml", "tests/unit")


@nox.session(name="integration")
def run_integration(session):
    """Run the integration tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
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
    # TODO: Temporary fix for Python 3.12 CI.
    # See: https://bitbucket.org/pybtex-devs/pybtex/issues/169/
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev,docs]", silent=False)
    session.run(
        "python",
        "-m",
        "pytest",
        "--doctest-plus",
        "pybamm",
    )


@nox.session(name="unit")
def run_unit(session):
    """Run the unit tests."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev,jax]", silent=False)
    if PYBAMM_ENV.get("PYBAMM_IDAKLU_EXPR_IREE") == "ON":
        # See comments in 'dev' session
        session.install(
            "-e",
            ".[iree]",
            "--find-links",
            PYBAMM_ENV.get("IREE_INDEX_URL"),
            silent=False,
        )
    session.run("python", "-m", "pytest", "-m", "unit")


@nox.session(name="examples")
def run_examples(session):
    """Run the examples tests for Jupyter notebooks."""
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev]", silent=False)
    notebooks_to_test = session.posargs if session.posargs else []
    session.run(
        "pytest", "--nbmake", *notebooks_to_test, "docs/source/examples/", external=True
    )


@nox.session(name="scripts")
def run_scripts(session):
    """Run the scripts tests for Python scripts."""
    set_environment_variables(PYBAMM_ENV, session=session)
    # Temporary fix for Python 3.12 CI. TODO: remove after
    # https://bitbucket.org/pybtex-devs/pybtex/issues/169/replace-pkg_resources-with
    # is fixed
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev]", silent=False)
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
    if PYBAMM_ENV.get("PYBAMM_IDAKLU_EXPR_IREE") == "ON":
        # Install IREE libraries for Jax-MLIR expression evaluation in the IDAKLU solver
        # (optional). IREE is currently pre-release and relies on nightly jaxlib builds.
        # When upgrading Jax/IREE ensure that the following are compatible with each other:
        #  - Jax and Jaxlib version [pyproject.toml]
        #  - IREE repository clone (use the matching nightly candidate) [noxfile.py]
        #  - IREE compiler matches Jaxlib (use the matching nightly build) [pyproject.toml]
        components.append("iree")
        args = ["--find-links", PYBAMM_ENV.get("IREE_INDEX_URL")]
    # Temporary fix for Python 3.12 CI. TODO: remove after
    # https://bitbucket.org/pybtex-devs/pybtex/issues/169/replace-pkg_resources-with
    # is fixed
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
    session.install("setuptools", silent=False)
    session.install("-e", ".[all,dev,jax]", silent=False)
    session.run("python", "-m", "pytest", "-m", "unit or integration")


@nox.session(name="docs")
def build_docs(session):
    """Build the documentation and load it in a browser tab, rebuilding on changes."""
    envbindir = session.bin
    # TODO: Temporary fix for Python 3.12 CI.
    # See: https://bitbucket.org/pybtex-devs/pybtex/issues/169/
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
