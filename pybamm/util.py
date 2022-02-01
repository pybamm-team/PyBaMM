#
# Utility classes for PyBaMM
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import argparse
import importlib.util
import numbers
import os
import pathlib
import pickle
import subprocess
import sys
import timeit
import warnings
from collections import defaultdict
from platform import system

import numpy as np
import pkg_resources

import pybamm

# versions of jax and jaxlib compatible with PyBaMM
JAX_VERSION = "0.2.12"
JAXLIB_VERSION = "0.1.70"


def root_dir():
    """return the root directory of the PyBaMM install directory"""
    return str(pathlib.Path(pybamm.__path__[0]).parent)


class FuzzyDict(dict):
    def levenshtein_ratio(self, s, t):
        """
        Calculates levenshtein distance between two strings s and t.
        Uses the formula from
        https://www.datacamp.com/community/tutorials/fuzzy-string-python
        """
        # Initialize matrix of zeros
        rows = len(s) + 1
        cols = len(t) + 1
        distance = np.zeros((rows, cols), dtype=int)

        # Populate matrix of zeros with the indices of each character of both strings
        for i in range(1, rows):
            for k in range(1, cols):
                distance[i][0] = i
                distance[0][k] = k

        # Iterate over the matrix to compute the cost of deletions, insertions and/or
        # substitutions
        for col in range(1, cols):
            for row in range(1, rows):
                if s[row - 1] == t[col - 1]:
                    # If the characters are the same in the two strings in a given
                    # position [i,j] then the cost is 0
                    cost = 0
                else:
                    # In order to align the results with those of the Python Levenshtein
                    # package, the cost of a substitution is 2.
                    cost = 2
                distance[row][col] = min(
                    distance[row - 1][col] + 1,  # Cost of deletions
                    distance[row][col - 1] + 1,  # Cost of insertions
                    distance[row - 1][col - 1] + cost,  # Cost of substitutions
                )
        # Computation of the Levenshtein Distance Ratio
        ratio = ((len(s) + len(t)) - distance[row][col]) / (len(s) + len(t))
        return ratio

    def get_best_matches(self, key):
        """Get best matches from keys"""
        key = key.lower()
        best_three = []
        lowest_score = 0
        for k in self.keys():
            score = self.levenshtein_ratio(k.lower(), key)
            # Start filling out the list
            if len(best_three) < 3:
                best_three.append((k, score))
                # Sort once the list has three elements, using scores
                if len(best_three) == 3:
                    best_three.sort(key=lambda x: x[1], reverse=True)
                    lowest_score = best_three[-1][1]
            # Once list is full, start checking new entries
            else:
                if score > lowest_score:
                    # Replace last element with new entry
                    best_three[-1] = (k, score)
                    # Sort and update lowest score
                    best_three.sort(key=lambda x: x[1], reverse=True)
                    lowest_score = best_three[-1][1]

        return [x[0] for x in best_three]

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if "negative electrode sei" in key.lower():
                raise KeyError(
                    f"'{key}' not found. All SEI parameters have been "
                    "renamed from '...negative electrode SEI...' to '...SEI...'"
                )
            if "negative electrode lithium plating" in key.lower():
                raise KeyError(
                    f"'{key}' not found. All lithium plating parameters have been "
                    "renamed from '...negative electrode lithium plating...' "
                    "to '...lithium plating...'"
                )
            best_matches = self.get_best_matches(key)
            raise KeyError(f"'{key}' not found. Best matches are {best_matches}")

    def search(self, key, print_values=False):
        """
        Search dictionary for keys containing 'key'. If print_values is True, then
        both the keys and values will be printed. Otherwise just the values will
        be printed. If no results are found, the best matches are printed.
        """
        key = key.lower()

        # Sort the keys so results are stored in alphabetical order
        keys = list(self.keys())
        keys.sort()
        results = {}

        # Check if any of the dict keys contain the key we are searching for
        for k in keys:
            if key in k.lower():
                results[k] = self[k]

        if results == {}:
            # If no results, return best matches
            best_matches = self.get_best_matches(key)
            print(
                f"No results for search using '{key}'. Best matches are {best_matches}"
            )
        elif print_values:
            # Else print results, including dict items
            print("\n".join("{}\t{}".format(k, v) for k, v in results.items()))
        else:
            # Just print keys
            print("\n".join("{}".format(k) for k in results.keys()))


class Timer(object):
    """
    Provides accurate timing.

    Example
    -------
    timer = pybamm.Timer()
    print(timer.time())

    """

    def __init__(self):
        self._start = timeit.default_timer()

    def reset(self):
        """
        Resets this timer's start time.
        """
        self._start = timeit.default_timer()

    def time(self):
        """
        Returns the time (float, in seconds) since this timer was created,
        or since meth:`reset()` was last called.
        """
        return TimerTime(timeit.default_timer() - self._start)


class TimerTime:
    def __init__(self, value):
        """A string whose value prints in human-readable form"""
        self.value = value

    def __str__(self):
        """
        Formats a (non-integer) number of seconds, returns a string like
        "5 weeks, 3 days, 1 hour, 4 minutes, 9 seconds", or "0.0019 seconds".
        """
        time = self.value
        if time < 1e-6:
            return "{:.3f} ns".format(time * 1e9)
        if time < 1e-3:
            return "{:.3f} us".format(time * 1e6)
        if time < 1:
            return "{:.3f} ms".format(time * 1e3)
        elif time < 60:
            return "{:.3f} s".format(time)
        output = []
        time = int(round(time))
        units = [(604800, "week"), (86400, "day"), (3600, "hour"), (60, "minute")]
        for k, name in units:
            f = time // k
            if f > 0 or output:
                output.append(str(f) + " " + (name if f == 1 else name + "s"))
            time -= f * k
        output.append("1 second" if time == 1 else str(time) + " seconds")
        return ", ".join(output)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(self.value + other)
        else:
            return TimerTime(self.value + other.value)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(self.value - other)
        else:
            return TimerTime(self.value - other.value)

    def __rsub__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(other - self.value)

    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(self.value * other)
        else:
            return TimerTime(self.value * other.value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(self.value / other)
        else:
            return TimerTime(self.value / other.value)

    def __rtruediv__(self, other):
        if isinstance(other, numbers.Number):
            return TimerTime(other / self.value)

    def __eq__(self, other):
        return self.value == other.value


def load_function(filename):
    """
    Load a python function from an absolute or relative path using `importlib`.
    Example - pybamm.load_function("pybamm/input/example.py")

    Arguments
    ---------
    filename : str
        The path of the file containing the function.

    Returns
    -------
    function
        The python function loaded from the file.
    """
    # Remove `.py` from the file name
    if filename.endswith(".py"):
        filename = filename.replace(".py", "")

    # Replace `lead-acid` with `lead_acid`
    if "lead-acid" in filename:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "lead-acid is deprecated, use lead_acid instead", DeprecationWarning
        )
        filename = filename.replace("lead-acid", "lead_acid")

    # Replace `lithium-ion` with `lithium_ion`
    if "lithium-ion" in filename:
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(
            "lithium-ion is deprecated, use lithium_ion instead", DeprecationWarning
        )
        filename = filename.replace("lithium-ion", "lithium_ion")

    # Assign path to _ and filename to tail
    _, tail = os.path.split(filename)

    # Store the current working directory
    orig_dir = os.getcwd()

    # Strip absolute path to pybamm/input/example.py
    if "pybamm" in filename:
        root_path = filename[filename.rfind("pybamm") :]
    # If the function is in the current working directory
    elif os.getcwd() in filename:
        root_path = filename.replace(os.getcwd(), "")
        # getcwd() returns "C:\\" when in the root drive and "C:\\a\\b\\c" otherwise
        if root_path[0] == "\\" or root_path[0] == "/":
            root_path = root_path[1:]
    # If the function is not in the current working directory and the path provided is
    # absolute
    elif os.path.isabs(filename) and not os.getcwd() in filename:  # pragma: no cover
        # Change directory to import the function
        dir_path = os.path.split(filename)[0]
        os.chdir(dir_path)
        root_path = filename.replace(os.getcwd(), "")
        root_path = root_path[1:]
    else:
        root_path = filename

    path = root_path.replace("/", ".")
    path = path.replace("\\", ".")
    pybamm.logger.debug(
        f"Importing function '{tail}' from file '{filename}' via path '{path}'"
    )
    module_object = importlib.import_module(path)

    # Revert back current working directory if it was changed
    os.chdir(orig_dir)
    return getattr(module_object, tail)


def rmse(x, y):
    """
    Calculate the root-mean-square-error between two vectors x and y, ignoring NaNs
    """
    # Check lengths
    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")
    return np.sqrt(np.nanmean((x - y) ** 2))


def get_infinite_nested_dict():
    """
    Return a dictionary that allows infinite nesting without having to define level by
    level.

    See:
    https://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python/652226#652226

    Example
    -------
    >>> import pybamm
    >>> d = pybamm.get_infinite_nested_dict()
    >>> d["a"] = 1
    >>> d["a"]
    1
    >>> d["b"]["c"]["d"] = 2
    >>> d["b"]["c"] == {"d": 2}
    True
    """
    return defaultdict(get_infinite_nested_dict)


def load(filename):
    """Load a saved object"""
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_parameters_filepath(path):
    """Returns path if it exists in current working dir,
    otherwise get it from package dir"""
    if os.path.exists(path):
        return path
    else:
        return os.path.join(pybamm.__path__[0], path)


def have_jax():
    """Check if jax and jaxlib are installed with the correct versions"""
    return (
        (importlib.util.find_spec("jax") is not None)
        and (importlib.util.find_spec("jaxlib") is not None)
        and is_jax_compatible()
    )


def is_jax_compatible():
    """Check if the available version of jax and jaxlib are compatible with PyBaMM"""
    return (
        pkg_resources.get_distribution("jax").version == JAX_VERSION
        and pkg_resources.get_distribution("jaxlib").version == JAXLIB_VERSION
    )


def install_jax(arguments=None):  # pragma: no cover
    """
    Install compatible versions of jax, jaxlib.

    Command Line Interface:
    -----------------------
    >>> pybamm_install_jax

    optional arguments:
    -h, --help   show help message
    -f, --force  force install compatible versions of jax and jaxlib
    """
    parser = argparse.ArgumentParser(description="Install jax and jaxlib")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force install compatible versions of"
        f" jax ({JAX_VERSION}) and jaxlib ({JAXLIB_VERSION})",
    )

    args = parser.parse_args(arguments)

    if system() == "Windows":
        raise NotImplementedError("Jax is not available on Windows")

    # Raise an error if jax and jaxlib are already installed, but incompatible
    # and --force is not set
    elif importlib.util.find_spec("jax") is not None:
        if not args.force and not is_jax_compatible():
            raise ValueError(
                "Jax is already installed but the installed version of jax or jaxlib is"
                " not supported by PyBaMM. \nYou can force install compatible versions"
                f" of jax ({JAX_VERSION}) and jaxlib ({JAXLIB_VERSION}) using the"
                " following command: \npybamm_install_jax --force"
            )

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"jax=={JAX_VERSION}",
            f"jaxlib=={JAXLIB_VERSION}",
        ]
    )
