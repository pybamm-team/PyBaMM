import nox
import os
import sys


@nox.session(name="pybamm-requires")
def run_pybamm_requires(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    if sys.platform != "win32" or sys.platform != "darwin":
        session.install("wget", "cmake")
        session.run("python", "scripts/install_KLU_Sundials.py")
        session.run(
            "git",
            "clone",
            "https://github.com/pybind/pybind11.git",
            "pybind11/",
            external=True,
        )


@nox.session(name="coverage")
def run_coverage(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    session.install("coverage")
    session.install("-e", ".[dev]")
    if sys.platform != "win32" or sys.platform != "darwin":
        session.install("scikits.odes")
        session.run("pybamm_install_jax")
    session.run("coverage", "run", "--rcfile=.coveragerc", "run-tests.py", "--nosub")
    session.run("coverage", "combine")
    session.run("coverage", "xml")


@nox.session(name="integration", reuse_venv=True)
def run_integration(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    session.install("-e", ".[dev]")
    if sys.platform == "linux":
        session.install("scikits.odes")
    session.run("python", "run-tests.py", "--integration")


@nox.session(name="doctests", reuse_venv=True)
def run_doctests(session):
    session.install("-e", ".[dev]")
    session.install("-e", ".[docs]")
    session.run("python", "run-tests.py", "--doctest")


@nox.session(name="unit", reuse_venv=True)
def run_unit(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    session.install("-e", ".[dev]")
    if sys.platform == "linux":
        session.run("pybamm_install_jax")
        session.install("scikits.odes")
    session.run("python", "run-tests.py", "--unit")


@nox.session(name="examples", reuse_venv=True)
def run_examples(session):
    session.install("-e", ".[dev]")
    session.install("-e", ".[examples]")
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
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        ">>",
        f"{envbindir}/activate",
    )


@nox.session(name="tests", reuse_venv=True)
def run_tests(session):
    session.install("-e", ".[dev]")
    if sys.platform == "linux":
        session.run("pybamm_install_jax")
    session.run("python", "run-tests.py", "--all")


@nox.session(name="docs", reuse_venv=True)
def build_docs(session):
    envbindir = session.bin
    session.install("-e", ".[dev]")
    session.install(
        "sphinx>=1.5",
        "pydata-sphinx-theme",
        "sphinx-autobuild",
        "sphinx_design",
        "sphinx-copybutton",
        "myst-parser",
        "sphinx-inline-tabs",
    )
    session.chdir("docs/")
    session.run(
        "sphinx-autobuild", "--open-browser", "-qT", ".", f"{envbindir}/../tmp/html"
    )
