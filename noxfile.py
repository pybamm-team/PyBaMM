import nox
import os
import sys

homedir = os.getenv("HOME")
PYBAMM_ENV = {
    "SUNDIALS_INST" : f"{homedir}/.local",
    "LD_LIBRARY_PATH" : f"{homedir}/.local/lib:"
}

def set_environment_variables(env_dict, session):
    """
    Sets environment variables for a nox session object.

    Args:
        session (nox.session): The session to set the environment
            variables for.
        env_dict (dict): A dictionary of environment variable names and values.

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
    session.install("-e", ".")
    if sys.platform != "win32":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("coverage", "run", "--rcfile=.coveragerc", "run-tests.py", "--nosub")
    session.run("coverage", "combine")
    session.run("coverage", "xml")


@nox.session(name="integration", reuse_venv=True)
def run_integration(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[dev]")
    if sys.platform == "linux":
        session.install("scikits.odes")
    session.run("python", "run-tests.py", "--integration")


@nox.session(name="doctests", reuse_venv=True)
def run_doctests(session):
    session.install("-e", ".[docs]")
    session.run("python", "run-tests.py", "--doctest")


@nox.session(name="unit", reuse_venv=True)
def run_unit(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".")
    if sys.platform == "linux":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("python", "run-tests.py", "--unit")


@nox.session(name="examples", reuse_venv=True)
def run_examples(session):
    session.install("-e", ".[dev]")
    session.run("python", "run-tests.py", "--examples")


@nox.session(name="dev", reuse_venv=True)
def set_dev(session):
    homedir = os.getenv("HOME")
    LD_LIBRARY_PATH = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    envbindir = session.bin
    session.install("-e", ".[dev]")
    session.install("cmake")
    session.run(
        "echo",
        "export",
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",  # noqa: F821
        ">>",
        f"{envbindir}/activate",
    )


@nox.session(name="tests", reuse_venv=True)
def run_tests(session):
    set_environment_variables(PYBAMM_ENV, session=session)
    session.install("-e", ".[dev]")
    if sys.platform == "linux" or sys.platform == "darwin":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("python", "run-tests.py", "--all")


@nox.session(name="docs", reuse_venv=True)
def build_docs(session):
    envbindir = session.bin
    session.install("-e", ".[docs]")
    session.chdir("docs/")
    session.run(
        "sphinx-autobuild", "--open-browser", "-qT", ".", f"{envbindir}/../tmp/html"
    )
