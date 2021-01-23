import os
import pybamm
import argparse
import shutil

from pathlib import Path


def yes_or_no(question):
    while "Please answer yes(y) or no(n).":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False


def get_parser(description):
    """
    Create a parser with the following arguments:
    parameter_dir: str
    battery_type: str
    component: str
    force: bool, optional

    Parameter:
    ----------
    description: A description of the command [str]

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "parameter_dir", type=str, help="Name of the parameter directory"
    )
    parser.add_argument("battery_type", choices=["lithium-ion", "lead-acid"])
    parser.add_argument(
        "component",
        choices=[
            "cells",
            "electrolytes",
            "experiments",
            "negative_electrodes",
            "positive_electrodes",
            "separators",
        ],
    )
    parser.add_argument("-f", "--force", action="store_true")
    return parser


def copy_directory(source, destination, overwrite):
    """
    Copy directory structure source as destination, dealing
    with overwriting.

    Parameters:
    -----------
    source: Directory structure [str]
    destination: Directory structure [str]
    overwrite: Whether or not to overwrite [bool]

    Example:
    --------
    source = "site-packages/pybamm/input/parameters/lead-acid/negative_electrodes/blah"
    destination = "input/parameters/lead-acid/negative_electrodes/blah"
    Will copy
      "input/parameters/lead-acid/negative_electrodes/blah"
    in current working directory.
    """
    try:
        shutil.copytree(source, destination)
    except FileExistsError:
        if overwrite:
            shutil.rmtree(destination)
            shutil.copytree(source, destination)
        elif yes_or_no("Parameter set already defined, erase?"):
            shutil.rmtree(destination)
            shutil.copytree(source, destination)
        else:
            print("Doing nothing.")
    # Implementation note:
    # In python 3.7 shutil.copytree() does not provide an option to
    # automatically overwrite a directory tree.
    # When switching to python 3.8, the following function could be
    # written using
    # shutil.copytree(args.parameter_dir, destination_dir, exist_ok=True)
    # and getting rid of the rmtree calls.
    # see https://docs.python.org/3/library/shutil.html


def add_parameter(arguments=None):
    """
    Add a parameter directory to package input directory.
    This allows the parameters to be used from anywhere in the system.

    Example:
    "add_parameter foo lithium-ion negative_electrodes" will copy directory foo in
    "pybamm/input/parameters/lithium-ion/negative_electrodes".
    """
    parser = get_parser("Copy parameter to the PyBaMM package directory.")
    args = parser.parse_args(arguments)

    parameters_root_dir = os.path.join(pybamm.__path__[0], "input", "parameters")

    parameter_dir_name = Path(args.parameter_dir).name
    destination_dir = os.path.join(
        parameters_root_dir, args.battery_type, args.component, parameter_dir_name
    )

    copy_directory(args.parameter_dir, destination_dir, args.force)


def remove_parameter(arguments=None):
    """
    Remove a parameter directory from package input directory.

    Example:
    "rm_parameter foo lithium-ion negative_electrodes" will remove directory foo in
    "pybamm/input/parameters/lithium-ion/negative_electrodes".
    """
    parser = get_parser("Remove parameters from the PyBaMM package directory.")
    args = parser.parse_args(arguments)

    parameters_root_dir = os.path.join(pybamm.__path__[0], "input", "parameters")

    parameter_dir_name = Path(args.parameter_dir).name
    destination_dir = os.path.join(
        parameters_root_dir, args.battery_type, args.component, parameter_dir_name
    )

    if not args.force:
        yes_or_no("This will remove directory {}, continue?".format(destination_dir))
    shutil.rmtree(destination_dir, ignore_errors=True)


def edit_parameter(arguments=None):
    """
    Copy a given default parameter directory to the current working directory
    for editing. For example

    .. code::

      edit_param(["lithium-ion"])

    will create the directory structure::

      lithium-ion/
        negative_electrodes/
          graphite_Chen2020
          ...
        positive_electrodes/
        ...

    in the current working directory.
    """
    desc = "Pull parameter directory dir to current working directory for editing."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("chemistry", choices=["lithium-ion", "lead-acid"])
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args(arguments)

    path = os.path.join("input", "parameters", args.chemistry)

    source_dir = os.path.join(pybamm.__path__[0], path)
    copy_directory(source_dir, args.chemistry, args.force)
