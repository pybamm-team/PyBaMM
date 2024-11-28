import importlib.util
import importlib.metadata
import numbers
import os
import pathlib
import pickle
import subprocess
import timeit
import difflib
from warnings import warn

import pybamm

# Versions of jax and jaxlib compatible with PyBaMM. Note: these are also defined in
# the extras dependencies in pyproject.toml, and therefore must be kept in sync.
JAX_VERSION = "0.4.27"
JAXLIB_VERSION = "0.4.27"


def root_dir():
    """return the root directory of the PyBaMM install directory"""
    return str(pathlib.Path(pybamm.__path__[0]).parent.parent)


def get_git_commit_info():
    """
    Get the git commit info for the current PyBaMM version, e.g. v22.8-39-gb25ce8c41
    (version 22.8, commit b25ce8c41)
    """
    try:
        # Get the latest git commit hash
        return str(
            subprocess.check_output(["git", "describe", "--tags"], cwd=root_dir())
            .strip()
            .decode()
        )
    except subprocess.CalledProcessError:  # pragma: no cover
        # Not a git repository so just return the version number
        return f"v{pybamm.__version__}"


class FuzzyDict(dict):
    def get_best_matches(self, key):
        """Get best matches from keys"""
        return difflib.get_close_matches(key, list(self.keys()), n=3, cutoff=0.5)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as error:
            if "electrode diffusivity" in key or "particle diffusivity" in key:
                old_term, new_term = (
                    ("electrode", "particle")
                    if "electrode diffusivity" in key
                    else ("particle", "electrode")
                )
                alternative_key = key.replace(old_term, new_term)

                if old_term == "electrode":
                    warn(
                        f"The parameter '{alternative_key}' has been renamed to '{key}' and will be removed in a future release. Using '{key}'",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                return super().__getitem__(alternative_key)
            if key in ["Negative electrode SOC", "Positive electrode SOC"]:
                domain = key.split(" ")[0]
                raise KeyError(
                    f"Variable '{domain} electrode SOC' has been renamed to "
                    f"'{domain} electrode stoichiometry' to avoid confusion "
                    "with cell SOC"
                ) from error
            if "Measured open circuit voltage" in key:
                raise KeyError(
                    "The variable that used to be called "
                    "'Measured open circuit voltage [V]' is now called "
                    "'Surface open-circuit voltage [V]'. There is also another "
                    "variable called 'Bulk open-circuit voltage [V]' which is the"
                    "open-circuit voltage evaluated at the average particle "
                    "concentrations."
                ) from error
            if "Open-circuit voltage at 0% SOC [V]" in key:
                raise KeyError(
                    "Parameter 'Open-circuit voltage at 0% SOC [V]' not found."
                    "In most cases this should be set to be equal to "
                    "'Lower voltage cut-off [V]'"
                ) from error
            if "Open-circuit voltage at 100% SOC [V]" in key:
                raise KeyError(
                    "Parameter 'Open-circuit voltage at 100% SOC [V]' not found."
                    "In most cases this should be set to be equal to "
                    "'Upper voltage cut-off [V]'"
                ) from error
            best_matches = self.get_best_matches(key)
            for k in best_matches:
                if key in k and k.endswith("]"):
                    raise KeyError(
                        f"'{key}' not found. Use the dimensional version '{k}' instead."
                    ) from error
            raise KeyError(
                f"'{key}' not found. Best matches are {best_matches}"
            ) from error

    def search(self, key, print_values=False):
        """
        Search dictionary for keys containing 'key'. If print_values is True, then
        both the keys and values will be printed. Otherwise just the values will
        be printed. If no results are found, the best matches are printed.
        """
        key_in = key
        key = key_in.lower()

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
                f"No results for search using '{key_in}'. "
                f"Best matches are {best_matches}"
            )
        elif print_values:
            # Else print results, including dict items
            print("\n".join(f"{k}\t{v}" for k, v in results.items()))
        else:
            # Just print keys
            print("\n".join(f"{k}" for k in results.keys()))

    def copy(self):
        return FuzzyDict(super().copy())


class Timer:
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
            return f"{time * 1e9:.3f} ns"
        if time < 1e-3:
            return f"{time * 1e6:.3f} us"
        if time < 1:
            return f"{time * 1e3:.3f} ms"
        elif time < 60:
            return f"{time:.3f} s"
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

    def __repr__(self):
        return f"pybamm.TimerTime({self.value})"

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


def has_jax():
    """
    Check if jax and jaxlib are installed with the correct versions

    Returns
    -------
    bool
        True if jax and jaxlib are installed with the correct versions, False if otherwise

    """
    return (
        (importlib.util.find_spec("jax") is not None)
        and (importlib.util.find_spec("jaxlib") is not None)
        and is_jax_compatible()
    )


def is_jax_compatible():
    """
    Check if the available versions of jax and jaxlib are compatible with PyBaMM

    Returns
    -------
    bool
        True if jax and jaxlib are compatible with PyBaMM, False if otherwise
    """
    return importlib.metadata.distribution("jax").version.startswith(
        JAX_VERSION
    ) and importlib.metadata.distribution("jaxlib").version.startswith(JAXLIB_VERSION)


def is_constant_and_can_evaluate(symbol):
    """
    Returns True if symbol is constant and evaluation does not raise any errors.
    Returns False otherwise.
    An example of a constant symbol that cannot be "evaluated" is PrimaryBroadcast(0).
    """
    if symbol.is_constant():
        try:
            symbol.evaluate()
            return True
        except NotImplementedError:
            return False
    else:
        return False


# https://docs.pybamm.org/en/latest/source/user_guide/contributing.html#managing-optional-dependencies-and-their-imports
def import_optional_dependency(module_name, attribute=None):
    err_msg = f"Optional dependency {module_name} is not available. See https://docs.pybamm.org/en/latest/source/user_guide/installation/index.html#optional-dependencies for more details."
    try:
        module = importlib.import_module(module_name)
        if attribute:
            if hasattr(module, attribute):
                imported_attribute = getattr(module, attribute)
                # Return the imported attribute
                return imported_attribute
            else:
                raise ModuleNotFoundError(err_msg)  # pragma: no cover
        else:
            # Return the entire module if no attribute is specified
            return module

    except ModuleNotFoundError as error:
        # Raise an ModuleNotFoundError if the module or attribute is not available
        raise ModuleNotFoundError(err_msg) from error
