#!/usr/bin/env python
#
# Runs all unit tests included in PyBaMM.
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
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
        print(f"Running {folder} tests with executable '{interpreter}'")
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


def run_scripts(executable="python"):
    """
    Run example scripts tests. Exits if they fail.
    """

    # Scan and run
    print("Testing scripts with executable `" + str(executable) + "`")

    # Test scripts in examples
    # TODO: add scripts to docs/source/examples
    if not scan_for_scripts("examples", True, executable):
        print("\nErrors encountered in scripts")
        sys.exit(1)
    print("\nOK")


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
    cmd = [executable, path]
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
    print(f"ok ({b.time()})")
    return True


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
    # Example notebooks tests
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Test all Jupyter notebooks in `docs/source/examples/` (deprecated, use nox or pytest instead).",
    )
    parser.add_argument(
        "--debook",
        nargs=2,
        metavar=("in", "out"),
        help="Export a Jupyter notebook to a Python file for manual testing.",
    )
    # Scripts tests
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Test all example scripts in `examples/`.",
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
    # Doctests
    if args.doctest:
        has_run = True
        run_doc_tests()
    # Notebook tests (deprecated)
    elif args.examples:
        raise ValueError(
            "Notebook tests are deprecated, use nox -s examples or pytest instead"
        )
    if args.debook:
        raise ValueError(
            "Notebook tests are deprecated, use nox -s examples or pytest instead"
        )
    # Scripts tests
    elif args.scripts:
        has_run = True
        run_scripts(interpreter)
    # Combined test sets
    if args.quick:
        has_run = True
        run_code_tests("all", interpreter=interpreter)
        run_doc_tests()
    # Help
    if not has_run:
        parser.print_help()
