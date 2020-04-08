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
            "anodes",
            "cathodes",
            "cells",
            "electrolytes",
            "experiments",
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
    source = "site-packages/pybamm/input/parameters/lead-acid/anodes/blablah"
    destination = "input/parameters/lead-acid/anodes/blablah"
    Will copy
      "input/parameters/lead-acid/anodes/blablah"
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
    "add_parameter foo lithium-ion anodes" will copy directory foo in
    "pybamm/input/parameters/lithium-ion/anodes".
    """
    parser = get_parser("Copy parameter to the PyBaMM package directory.")
    args = parser.parse_args(arguments)

    parameters_root_dir = os.path.join(pybamm.__path__[0], "input", "parameters")

    parameter_dir_name = Path(args.parameter_dir).name
    destination_dir = os.path.join(
        parameters_root_dir, args.battery_type, args.component, parameter_dir_name
    )

    copy_directory(args.parameter_dir, destination_dir, args.force)
    print("Copied {} to {}".format(args.parameter_dir, destination_dir))


def edit_parameter(arguments=None):
    """
    Copy a given parameter package directory to the current working directory
    for editing. The copy preserves the directory structure within the "input"
    directory, i.e

    ``edit_param(["graphite_Kim2011","lithium-ion","anodes"])``

    will create the directory structure
    "input/parameters/lithium-ion/anodes/graphite_Kim2011"
    in the current working directory.
    """
    parser = get_parser(
        "Pull parameter directory dir to current working directory for editing."
    )
    args = parser.parse_args(arguments)

    path = os.path.join(
        "input", "parameters", args.battery_type, args.component, args.parameter_dir
    )

    source_dir = os.path.join(pybamm.__path__[0], path)
    copy_directory(source_dir, path, args.force)


def list_parameters(arguments=None):
    """
    Output a list of available parameter sets for a given chemistry and component.
    The list is divided into package parameter serts and local parameter sets,
    located in the current working directory.

    >>> from pybamm.parameters_cli import list_parameters
    >>> list_parameters(["lithium-ion", "anodes"])
    Available package parameters:
      * graphite_Ecker2015
      * graphite_Chen2020
      * graphite_mcmb2528_Marquis2019
      * graphite_UMBL_Mohtat2020
      * graphite_Kim2011
    Available local parameters:
    """
    parser = argparse.ArgumentParser(
        description="List available parameter sets for a given chemistry and component."
    )
    parser.add_argument("battery_type", choices=["lithium-ion", "lead-acid"])
    parser.add_argument(
        "component",
        choices=[
            "anodes",
            "cathodes",
            "cells",
            "electrolytes",
            "experiments",
            "separators",
        ],
    )

    args = parser.parse_args(arguments)

    package_dir = os.path.join(
        pybamm.__path__[0], "input", "parameters", args.battery_type, args.component
    )
    root, package_dirs, files = next(os.walk(package_dir))

    print("Available package parameters:")
    for dirname in package_dirs:
        print("  * {}".format(dirname))

    local_dir = os.path.join("input", "parameters", args.battery_type, args.component)
    if os.path.isdir(local_dir):
        root, local_dirs, files = next(os.walk(local_dir))
    else:
        local_dirs = []
    print("Available local parameters:")
    for dirname in local_dirs:
        print("  * {}".format(dirname))
