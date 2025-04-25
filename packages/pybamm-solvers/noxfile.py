import nox
import os
import sys
import warnings
import platform
from pathlib import Path


nox.options.default_venv_backend = "virtualenv"
nox.options.reuse_existing_virtualenvs = True
if sys.platform != "win32":
    nox.options.sessions = ["idaklu-requires", "unit"]
else:
    nox.options.sessions = ["unit"]


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


homedir = Path(__file__)
PYBAMM_ENV = {
    "LD_LIBRARY_PATH": f"{homedir}/.idaklu/lib",
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


@nox.session(name="idaklu-requires")
def run_pybamm_requires(session):
    """Download, compile, and install the build-time requirements for Linux and macOS.
    Supports --install-dir for custom installation paths and --force to force installation."""
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        session.run("python", "install_KLU_Sundials.py", *session.posargs)
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
        session.error("nox -s idaklu-requires is only available on Linux & macOS.")


@nox.session(name="unit")
def run_unit(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("setuptools", silent=False)
    session.install("casadi==3.6.7", silent=False)
    session.install(".[dev]", silent=False)
    if PYBAMM_ENV.get("PYBAMM_IDAKLU_EXPR_IREE") == "ON":
        # See comments in 'dev' session
        session.install(
            ".[iree]",
            "--find-links",
            PYBAMM_ENV.get("IREE_INDEX_URL"),
            silent=False,
        )
    session.run("pytest", "tests")
