#!/usr/bin/env python
#
# Runs all unit tests included in PyBaMM.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import re
import os
import shutil
import pybamm
import sys
import argparse
import unittest
import subprocess


def run_code_tests(executable=False, folder: str = "unit", interpreter="python"):
    """
    Runs tests, exits if they don't finish.
    Parameters
    ----------
    executable : bool (default False)
        If True, tests are run in subprocesses using the executable 'python'.
        Must be True for travis tests (otherwise tests always 'pass')
    folder : str
        Which folder to run the tests from (unit, integration or both ('all'))
    """
    if folder == "all":
        tests = "tests/"
    else:
        tests = "tests/" + folder
        if folder == "unit":
            pybamm.settings.debug_mode = True
    if interpreter == "python":
        # Make sure to refer to the interpreter for the
        # currently activated virtual environment
        interpreter = sys.executable
    if executable is False:
        suite = unittest.defaultTestLoader.discover(tests, pattern="test*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        ret = int(not result.wasSuccessful())
    else:
        print("Running {} tests with executable '{}'".format(folder, interpreter))
        cmd = [interpreter, "-m", "unittest", "discover", "-v", tests]
        p = subprocess.Popen(cmd)
        try:
            ret = p.wait()
        except KeyboardInterrupt:
            try:
                p.terminate()
            except OSError:
                pass
            p.wait()
            print("")
            sys.exit(1)

    if ret != 0:
        sys.exit(ret)


def run_doc_tests():
    """
    Checks if the documentation can be built, runs any doctests (currently not
    used).
    """
    print("Checking if docs can be built.")
    p = subprocess.Popen(
        [
            "sphinx-build",
            "-j",
            "auto",
            "-b",
            "doctest",
            "docs",
            "docs/build/html",
            "-W",
            "--keep-going",
        ]
    )
    try:
        ret = p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except OSError:
            pass
        p.wait()
        print("")
        sys.exit(1)
    if ret != 0:
        print("FAILED")
        sys.exit(ret)
    # delete the entire docs/source/build folder + files since it currently
    # causes problems with nbsphinx in further docs or doctest builds
    print("Deleting built files.")
    shutil.rmtree("docs/build")


def run_notebook_and_scripts(executable="python"):
    """
    Runs Jupyter notebook and example scripts tests. Exits if they fail.
    """

    # Scan and run
    print("Testing notebooks and scripts with executable `" + str(executable) + "`")

    # Test notebooks in docs/source/examples
    if not scan_for_notebooks("docs/source/examples", True, executable):
        print("\nErrors encountered in notebooks")
        sys.exit(1)

    # Test scripts in examples
    # TODO: add scripts to docs/source/examples
    if not scan_for_scripts("examples", True, executable):
        print("\nErrors encountered in scripts")
        sys.exit(1)
    print("\nOK")


def scan_for_notebooks(root, recursive=True, executable="python"):
    """
    Scans for, and tests, all notebooks in a directory.
    """
    ok = True
    debug = False

    # Scan path
    for filename in os.listdir(root):
        path = os.path.join(root, filename)

        # Recurse into subdirectories
        if recursive and os.path.isdir(path):
            # Ignore hidden directories
            if filename[:1] == ".":
                continue
            ok &= scan_for_notebooks(path, recursive, executable)

        # Test notebooks
        if os.path.splitext(path)[1] == ".ipynb":
            if debug:
                print(path)
            else:
                ok &= test_notebook(path, executable)

    # Return True if every notebook is ok
    return ok


def scan_for_scripts(root, recursive=True, executable="python"):
    """
    Scans for, and tests, all scripts in a directory.
    """
    ok = True
    debug = False

    # Scan path
    for filename in os.listdir(root):
        path = os.path.join(root, filename)

        # Recurse into subdirectories
        if recursive and os.path.isdir(path):
            # Ignore hidden directories
            if filename[:1] == ".":
                continue
            ok &= scan_for_scripts(path, recursive, executable)

        # Test scripts
        elif os.path.splitext(path)[1] == ".py":
            if debug:
                print(path)
            else:
                ok &= test_script(path, executable)

    # Return True if every script is ok
    return ok


def test_notebook(path, executable="python"):
    """
    Tests a single notebook, exits if it doesn't finish.
    """
    import nbconvert
    import pybamm

    b = pybamm.Timer()
    print("Test " + path + " ... ", end="")
    sys.stdout.flush()

    # Make sure the notebook has a "%pip install pybamm -q" command, for using Google
    # Colab
    with open(path, "r") as f:
        if "%pip install pybamm -q" not in f.read():
            # print error and exit
            print("\n" + "-" * 70)
            print("ERROR")
            print("-" * 70)
            print("Installation command '%pip install pybamm -q' not found in notebook")
            print("-" * 70)
            return False

    # Make sure the notebook has "pybamm.print_citations()" to print the relevant papers
    with open(path, "r") as f:
        if "pybamm.print_citations()" not in f.read():
            # print error and exit
            print("\n" + "-" * 70)
            print("ERROR")
            print("-" * 70)
            print(
                "Print citations command 'pybamm.print_citations()' not found in "
                "notebook"
            )
            print("-" * 70)
            return False

    # Load notebook, convert to Python
    e = nbconvert.exporters.PythonExporter()
    code, __ = e.from_filename(path)

    # Remove coding statement, if present
    code = "\n".join([x for x in code.splitlines() if x[:9] != "# coding"])

    # Tell matplotlib not to produce any figures
    env = dict(os.environ)
    env["MPLBACKEND"] = "Template"

    # If notebook makes use of magic commands then
    # the script must be run using ipython
    # https://github.com/jupyter/nbconvert/issues/503#issuecomment-269527834
    executable = (
        "ipython"
        if (("run_cell_magic(" in code) or ("run_line_magic(" in code))
        else executable
    )

    # Run in subprocess
    cmd = [executable] + ["-c", code]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        stdout, stderr = p.communicate()
        # TODO: Use p.communicate(timeout=3600) if Python3 only
        if p.returncode != 0:
            # Show failing code, output and errors before returning
            print("ERROR")
            print("-- script " + "-" * (79 - 10))
            for i, line in enumerate(code.splitlines()):
                j = str(1 + i)
                print(j + " " * (5 - len(j)) + line)
            print("-- stdout " + "-" * (79 - 10))
            print(str(stdout, "utf-8"))
            print("-- stderr " + "-" * (79 - 10))
            print(str(stderr, "utf-8"))
            print("-" * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print("ABORTED")
        sys.exit(1)

    # Sucessfully run
    print("ok ({})".format(b.time()))
    return True


def test_script(path, executable="python"):
    """
    Tests a single script, exits if it doesn't finish.
    """
    import pybamm

    b = pybamm.Timer()
    print("Test " + path + " ... ", end="")
    sys.stdout.flush()

    # Tell matplotlib not to produce any figures
    env = dict(os.environ)
    env["MPLBACKEND"] = "Template"

    # Run in subprocess
    cmd = [executable] + [path]
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
        )
        stdout, stderr = p.communicate()
        # TODO: Use p.communicate(timeout=3600) if Python3 only
        if p.returncode != 0:
            # Show failing code, output and errors before returning
            print("ERROR")
            print("-- stdout " + "-" * (79 - 10))
            print(str(stdout, "utf-8"))
            print("-- stderr " + "-" * (79 - 10))
            print(str(stderr, "utf-8"))
            print("-" * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print("ABORTED")
        sys.exit(1)

    # Sucessfully run
    print("ok ({})".format(b.time()))
    return True


def export_notebook(ipath, opath):
    """
    Exports the notebook at `ipath` to a Python file at `opath`.
    """
    import nbconvert
    from traitlets.config import Config

    # Create nbconvert configuration to ignore text cells
    c = Config()
    c.TemplateExporter.exclude_markdown = True

    # Load notebook, convert to Python
    e = nbconvert.exporters.PythonExporter(config=c)
    code, __ = e.from_filename(ipath)

    # Remove "In [1]:" comments
    r = re.compile(r"(\s*)# In\[([^]]*)\]:(\s)*")
    code = r.sub("\n\n", code)

    # Store as executable script file
    with open(opath, "w") as f:
        f.write("#!/usr/bin/env python")
        f.write(code)
    os.chmod(opath, 0o775)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Run unit tests for PyBaMM.",
        epilog="To run individual unit tests, use e.g. '$ tests/unit/test_timer.py'",
    )

    # Unit tests
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests using the python interpreter.",
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests using the `python` interpreter.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (unit and integration) using the `python` interpreter.",
    )
    parser.add_argument(
        "--nosub",
        action="store_true",
        help="Run unit tests without starting a subprocess.",
    )
    # Notebook tests
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Test all Jupyter notebooks and scripts in `examples`.",
    )
    parser.add_argument(
        "-debook",
        nargs=2,
        metavar=("in", "out"),
        help="Export a Jupyter notebook to a Python file for manual testing.",
    )
    # Flake8 (deprecated)
    parser.add_argument(
        "--flake8",
        action="store_true",
        help="Run flake8 to check for style issues (deprecated, use pre-commit)",
    )
    # Doctests
    parser.add_argument(
        "--doctest",
        action="store_true",
        help="Run any doctests, check if docs can be built",
    )
    # Combined test sets
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick checks (code tests, docs)",
    )
    # Non-standard Python interpreter name for subprocesses
    parser.add_argument(
        "--interpreter",
        nargs="?",
        default="python",
        metavar="python",
        help="Give the name of the Python interpreter if it is not 'python'",
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit vs integration
    interpreter = args.interpreter
    # Unit tests
    if args.integration:
        has_run = True
        run_code_tests(True, "integration", interpreter)
    if args.unit:
        has_run = True
        run_code_tests(True, "unit", interpreter)
    if args.all:
        has_run = True
        run_code_tests(True, "all", interpreter)
    if args.nosub:
        has_run = True
        run_code_tests(folder="unit", interpreter=interpreter)
    # Flake8
    if args.flake8:
        raise NotImplementedError("flake8 is no longer used. Use pre-commit instead.")
    # Doctests
    if args.doctest:
        has_run = True
        run_doc_tests()
    # Notebook tests
    elif args.examples:
        has_run = True
        run_notebook_and_scripts(interpreter)
    if args.debook:
        has_run = True
        export_notebook(*args.debook)
    # Combined test sets
    if args.quick:
        has_run = True
        run_code_tests("all", interpreter=interpreter)
        run_doc_tests()
    # Help
    if not has_run:
        parser.print_help()
