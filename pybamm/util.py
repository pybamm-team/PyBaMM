#
# Utility classes for PyBaMM
#
# The code in this file is adapted from Pints
# (see https://github.com/pints-team/pints)
#
import importlib
import numpy as np
import os
import sys
import timeit
import pathlib
import pickle
import pybamm
import numbers
from collections import defaultdict


def root_dir():
    """ return the root directory of the PyBaMM install directory """
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
    Load a python function from a file "function_name.py" called "function_name".
    The filename might either be an absolute path, in which case that specific file will
    be used, or the file will be searched for relative to PyBaMM root.

    Arguments
    ---------
    filename : str
        The name of the file containing the function of the same name.

    Returns
    -------
    function
        The python function loaded from the file.
    """

    if not filename.endswith(".py"):
        raise ValueError("Expected filename.py, but got {}".format(filename))

    # If it's an absolute path, find that exact file
    if os.path.isabs(filename):
        if not os.path.isfile(filename):
            raise ValueError(
                "{} is an absolute path, but the file is not found".format(filename)
            )

        valid_filename = filename

    # Else, search in the whole PyBaMM directory for matches
    else:
        search_path = pybamm.root_dir()

        head, tail = os.path.split(filename)

        matching_files = []

        for root, _, files in os.walk(search_path):
            for file in files:
                if file == tail:
                    full_path = os.path.join(root, file)
                    if full_path.endswith(filename):
                        matching_files.append(full_path)

        if len(matching_files) == 0:
            raise ValueError(
                "{} cannot be found in the PyBaMM directory".format(filename)
            )
        elif len(matching_files) > 1:
            raise ValueError(
                "{} found multiple times in the PyBaMM directory."
                "Consider using absolute file path.".format(filename)
            )

        valid_filename = matching_files[0]

    # Now: we have some /path/to/valid/filename.py
    # Add "/path/to/vaid" to the python path, and load the module "filename".
    # Then, check "filename" module contains "filename" function.  If it does, return
    # that function object, or raise an exception

    valid_path, valid_leaf = os.path.split(valid_filename)
    sys.path.append(valid_path)

    # Load the module, which must be the leaf of filename, minus the .py extension
    valid_module = valid_leaf.replace(".py", "")
    module_object = importlib.import_module(valid_module)

    # Check that a function of the same name exists in the loaded module
    if valid_module not in dir(module_object):
        raise ValueError(
            "No function {} found in module {}".format(valid_module, valid_module)
        )

    # Remove valid_path from sys_path to avoid clashes down the line
    sys.path.remove(valid_path)

    return getattr(module_object, valid_module)


def rmse(x, y):
    """Calculate the root-mean-square-error between
    two vectors x and y, ignoring NaNs"""
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
