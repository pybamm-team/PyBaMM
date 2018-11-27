#!/usr/bin/env python
#
# Runs all unit tests included in PyBaMM.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import re
import os
import sys
import argparse
import unittest
import subprocess


def run_unit_tests(executable=None):
    """
    Runs unit tests, exits if they don't finish.

    If an ``executable`` is given, tests are run in subprocesses using the
    given executable (e.g. ``python2`` or ``python3``).
    """
    tests = "tests/"
    if executable is None:
        suite = unittest.defaultTestLoader.discover(tests, pattern="test*.py")
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        print("Running unit tests with executable `" + executable + "`")
        cmd = [executable] + ["-m", "unittest", "discover", "-v", tests]
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


def run_flake8():
    """
    Runs flake8 in a subprocess, exits if it doesn't finish.
    """
    print("Running flake8 ... ")
    sys.stdout.flush()
    p = subprocess.Popen(["flake8"], stderr=subprocess.PIPE)
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
    if ret == 0:
        print("ok")
    else:
        print("FAILED")
        sys.exit(ret)


def run_doctests():
    """
    Checks if the documentation can be built, runs any doctests (currently not
    used).
    """
    print("Checking if docs can be built.")
    p = subprocess.Popen(
        ["sphinx-build", "-b", "doctest", "docs", "docs/build/html", "-W"]
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


def run_notebook_tests(skip_slow_books=False, executable="python"):
    """
    Runs Jupyter notebook tests. Exits if they fail.
    """
    # Ignore slow books?
    ignore_list = []
    if skip_slow_books and os.path.isfile(".slow-books"):
        with open(".slow-books", "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line or line[:1] == "#":
                    continue
                if not line.startswith("examples/"):
                    line = "examples/" + line
                if not line.endswith(".ipynb"):
                    line = line + ".ipynb"
                if not os.path.isfile(line):
                    raise Exception("Slow notebook note found: " + line)
                ignore_list.append(line)

    # Scan and run
    print("Testing notebooks with executable `" + str(executable) + "`")
    if not scan_for_notebooks("examples", True, executable, ignore_list):
        print("\nErrors encountered in notebooks")
        sys.exit(1)
    print("\nOK")


def scan_for_notebooks(root, recursive=True, executable="python", ignore_list=[]):
    """
    Scans for, and tests, all notebooks in a directory.
    """
    ok = True
    debug = False

    # Scan path
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if path in ignore_list:
            print("Skipping slow book: " + path)
            continue

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


def test_notebook(path, executable="python"):
    """
    Tests a single notebook, exists if it doesn't finish.
    """
    import nbconvert
    import pybamm

    b = pybamm.Timer()
    print("Test " + path + " ... ", end="")
    sys.stdout.flush()

    # Load notebook, convert to python
    e = nbconvert.exporters.PythonExporter()
    code, __ = e.from_filename(path)

    # Remove coding statement, if present
    code = "\n".join([x for x in code.splitlines() if x[:9] != "# coding"])

    # Tell matplotlib not to produce any figures
    env = dict(os.environ)
    env["MPLBACKEND"] = "Template"

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
            print(stdout)
            print("-- stderr " + "-" * (79 - 10))
            print(stderr)
            print("-" * 79)
            return False
    except KeyboardInterrupt:
        p.terminate()
        print("ABORTED")
        sys.exit(1)

    # Sucessfully run
    print("ok (" + b.format() + ")")
    return True


def export_notebook(ipath, opath):
    """
    Exports the notebook at `ipath` to a python file at `opath`.
    """
    import nbconvert
    from traitlets.config import Config

    # Create nbconvert configuration to ignore text cells
    c = Config()
    c.TemplateExporter.exclude_markdown = True

    # Load notebook, convert to python
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
        description="Run unit tests for Pints.",
        epilog="To run individual unit tests, use e.g."
        " $ pybamm/tests/test_simulation.py",
    )
    # Unit tests
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run all unit tests using the `python` interpreter.",
    )
    parser.add_argument(
        "--unit2",
        action="store_true",
        help="Run all unit tests using the `python2` interpreter.",
    )
    parser.add_argument(
        "--unit3",
        action="store_true",
        help="Run all unit tests using the `python3` interpreter.",
    )
    parser.add_argument(
        "--nosub",
        action="store_true",
        help="Run all unit tests without starting a subprocess.",
    )
    # Notebook tests
    parser.add_argument(
        "--books",
        action="store_true",
        help="Test only the fast Jupyter notebooks in `examples`.",
    )
    parser.add_argument(
        "--allbooks",
        action="store_true",
        help="Test all Jupyter notebooks in `examples`.",
    )
    parser.add_argument(
        "-debook",
        nargs=2,
        metavar=("in", "out"),
        help="Export a Jupyter notebook to a Python file for manual testing.",
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
        help="Run quick checks (unit tests, flake8, docs)",
    )

    # Parse!
    args = parser.parse_args()

    # Run tests
    has_run = False
    # Unit tests
    if args.unit:
        has_run = True
        run_unit_tests("python")
    if args.unit2:
        raise NotImplementedError
        has_run = True
        run_unit_tests("python2")
    if args.unit3:
        has_run = True
        run_unit_tests("python3")
    if args.nosub:
        has_run = True
        run_unit_tests()
    # Doctests
    if args.doctest:
        has_run = True
        run_doctests()
    # Notebook tests
    if args.allbooks:
        has_run = True
        run_notebook_tests()
    elif args.books:
        has_run = True
        run_notebook_tests(True)
    if args.debook:
        has_run = True
        export_notebook(*args.debook)
    # Combined test sets
    if args.quick:
        has_run = True
        run_flake8()
        run_unit_tests()
        run_doctests()
    # Help
    if not has_run:
        parser.print_help()
