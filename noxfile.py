import nox
import os
import sys

nox.options.sessions = ["pre-commit", "unit"]


@nox.session(name="pybamm-requires", reuse_venv=True)
def run_pybamm_requires(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
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
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
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
    session.install("-e", ".[docs]")
    session.run("python", "run-tests.py", "--doctest")


@nox.session(name="unit", reuse_venv=True)
def run_unit(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    session.install("-e", ".")
    if sys.platform == "linux":
        session.run("pybamm_install_jax")
        session.install("scikits.odes")
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
        f"LD_LIBRARY_PATH={LD_LIBRARY_PATH}",
        ">>",
        f"{envbindir}/activate",
    )


@nox.session(name="tests", reuse_venv=True)
def run_tests(session):
    homedir = os.getenv("HOME")
    session.env["SUNDIALS_INST"] = session.env.get("SUNDIALS_INST", f"{homedir}/.local")
    session.env[
        "LD_LIBRARY_PATH"
    ] = f"{homedir}/.local/lib:{session.env.get('LD_LIBRARY_PATH')}"
    session.install("-e", ".[dev]")
    if sys.platform == "linux" or sys.platform == "darwin":
        session.run("pybamm_install_jax")
        session.install("scikits.odes")
    session.run("python", "run-tests.py", "--all")


@nox.session(name="docs", reuse_venv=True)
def build_docs(session):
    envbindir = session.bin
    session.install("-e", ".[docs]")
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
