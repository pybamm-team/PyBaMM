import nox
import os
import sys

if sys.platform == "linux":
    nox.options.sessions = ["pre-commit", "pybamm-requires", "unit"]
else:
    nox.options.sessions = ["pre-commit", "unit"]


homedir = os.getenv("HOME")
PYBAMM_ENV = {
    "SUNDIALS_INST": f"{homedir}/.local",
    "LD_LIBRARY_PATH": f"{homedir}/.local/lib:",
}


def set_environment_variables(env_dict, session):
    """
    Sets environment variables for a nox session object.

    Parameters
    -----------
        session : nox.Session
            The session to set the environment variables for.
        env_dict : dict
            A dictionary of environment variable names and values.

    """
    for key, value in env_dict.items():
        session.env[key] = value


@nox.session(name="pybamm-requires", reuse_venv=True)
def run_pybamm_requires(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    if sys.platform != "win32":
        session.install("wget", "cmake")
        session.run("python", "scripts/install_KLU_Sundials.py")
        if not os.path.exists("./pybind11"):
            session.run(
                "git",
                "clone",
                "https://github.com/pybind/pybind11.git",
                "pybind11/",
                external=True,
            )
    else:
        session.error("nox -s pybamm-requires is only available on Linux & MacOS.")


@nox.session(name="coverage", reuse_venv=True)
def run_coverage(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("coverage")
    session.install("-e", ".[all]")
    if sys.platform != "win32":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("coverage", "run", "--rcfile=.coveragerc", "run-tests.py", "--nosub")
    session.run("coverage", "combine")
    session.run("coverage", "xml")


@nox.session(name="integration", reuse_venv=True)
def run_integration(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all]")
    if sys.platform == "linux":
        session.install("scikits.odes")
    session.run("python", "run-tests.py", "--integration")


@nox.session(name="doctests", reuse_venv=True)
def run_doctests(session):
    session.install("-e", ".[all,docs]")
    session.run("python", "run-tests.py", "--doctest")


@nox.session(name="unit", reuse_venv=True)
def run_unit(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all]")
    if sys.platform == "linux":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("python", "run-tests.py", "--unit")


@nox.session(name="examples", reuse_venv=True)
def run_examples(session):
    session.install("-e", ".[all]")
    session.run("python", "run-tests.py", "--examples")


@nox.session(name="dev", reuse_venv=True)
def set_dev(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    envbindir = session.bin
    session.install("-e", ".[all]")
    session.install("cmake")
    session.run(
        "echo",
        "export",
        f"LD_LIBRARY_PATH={PYBAMM_ENV['LD_LIBRARY_PATH']}",
        ">>",
        f"{envbindir}/activate",
    )


@nox.session(name="tests", reuse_venv=True)
def run_tests(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[all]")
    if sys.platform == "linux" or sys.platform == "darwin":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("python", "run-tests.py", "--all")


@nox.session(name="docs", reuse_venv=True)
def build_docs(session):
    envbindir = session.bin
    session.install("-e", ".[all,docs]")
    with session.chdir("docs/"):
        session.run(
            "sphinx-autobuild",
            "-j",
            "auto",
            "--open-browser",
            "-qT",
            ".",
            f"{envbindir}/../tmp/html",
        )


@nox.session(name="pre-commit", reuse_venv=True)
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
