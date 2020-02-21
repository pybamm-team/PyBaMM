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


def main(arguments=None):
    parser = argparse.ArgumentParser(description="Add a new parameter set")
    parser.add_argument(
        "parameter_dir",
        type=str,
        help="a str containing the name of the parameter directory",
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

    args = parser.parse_args(arguments)

    open(os.path.join(args.parameter_dir, "parameters.csv"))

    parameters_root_dir = os.path.join(pybamm.__path__[0], "input/parameters")

    parameter_dir_name = Path(args.parameter_dir).name
    destination_dir = os.path.join(
        parameters_root_dir, args.battery_type, args.component, parameter_dir_name
    )

    try:
        shutil.copytree(args.parameter_dir, destination_dir)
    except FileExistsError:
        if args.force:
            shutil.rmtree(destination_dir)
            shutil.copytree(args.parameter_dir, destination_dir)
        elif yes_or_no("Parameter set already defined, erase?"):
            shutil.rmtree(destination_dir)
            shutil.copytree(args.parameter_dir, destination_dir)
        else:
            print("Doing nothing.")
