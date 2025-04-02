from __future__ import annotations
import importlib.util
import importlib.metadata
import numbers
import os
import pathlib
import pickle
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
                if key in k and k.endswith("]") and not key.endswith("]"):
                    raise KeyError(
                        f"'{key}' not found. Use the dimensional version '{k}' instead."
                    ) from error
                elif key in k and (
                    k.startswith("Primary") or k.startswith("Secondary")
                ):
                    raise KeyError(
                        f"'{key}' not found. If you are using a composite model, you may need to use {k} instead. Otherwise, best matches are {best_matches}"
                    ) from error
            raise KeyError(
                f"'{key}' not found. Best matches are {best_matches}"
            ) from error

    def _find_matches(
        self, search_key: str, known_keys: list[str], min_similarity: float = 0.4
    ):
        """
        Helper method to find exact and partial matches for a given search key.

        Parameters
        ----------
        search_key : str
            The term to search for in the keys.
        known_keys : list of str
            The list of known dictionary keys to search within.
        min_similarity : float, optional
            The minimum similarity threshold for a match.
            Default is 0.4
        """
        search_key = search_key.lower()
        exact_matches = []
        partial_matches = []

        for key in known_keys:
            key_lower = key.lower()
            if search_key in key_lower:
                key_words = key_lower.split()

                for word in key_words:
                    similarity = difflib.SequenceMatcher(None, search_key, word).ratio()

                    if similarity >= min_similarity:
                        exact_matches.append(key)

            else:
                partial_matches = difflib.get_close_matches(
                    search_key, known_keys, n=5, cutoff=0.5
                )
        return exact_matches, partial_matches

    def search(
        self,
        keys: str | list[str],
        print_values: bool = False,
        min_similarity: float = 0.4,
    ):
        """
        Search dictionary for keys containing all terms in 'keys'.
        If print_values is True, both the keys and values will be printed.
        Otherwise, just the keys will be printed. If no results are found,
        the best matches are printed.

        Parameters
        ----------
        keys : str or list of str
            Search term(s)
        print_values : bool, optional
            If True, print both keys and values. Otherwise, print only keys.
            Default is False.
        min_similarity : float, optional
            The minimum similarity threshold for a match.
            Default is 0.4
        """

        if not isinstance(keys, (str, list)) or not all(
            isinstance(k, str) for k in keys
        ):
            msg = f"'keys' must be a string or a list of strings, got {type(keys)}"
            raise TypeError(msg)

        if isinstance(keys, str):
            if not keys.strip():
                msg = "The search term cannot be an empty or whitespace-only string"
                raise ValueError(msg)
            original_keys = [keys]
            search_keys = [keys.strip().lower()]

        elif isinstance(keys, list):
            if all(not str(k).strip() for k in keys):
                msg = "The 'keys' list cannot contain only empty or whitespace strings"
                raise ValueError(msg)

            original_keys = keys
            search_keys = [k.strip().lower() for k in keys if k.strip()]

        known_keys = list(self.keys())
        # Check for exact matches where all search keys appear together in a key
        exact_matches = []
        for key in known_keys:
            key_lower = key.lower()
            if all(term in key_lower for term in search_keys):
                key_words = key_lower.split()

                # Ensure all search terms match at least one word in the key
                if all(
                    any(
                        difflib.SequenceMatcher(None, term, word).ratio()
                        >= min_similarity
                        for word in key_words
                    )
                    for term in search_keys
                ):
                    exact_matches.append(key)

        if exact_matches:
            print(
                f"Results for '{' '.join(k for k in original_keys if k.strip())}': {exact_matches}"
            )
            if print_values:
                for match in exact_matches:
                    print(f"{match} -> {self[match]}")
            return

        # If no exact matches, iterate over search keys individually
        for original_key, search_key in zip(original_keys, search_keys):
            exact_key_matches, partial_matches = self._find_matches(
                search_key, known_keys, min_similarity
            )

            if exact_key_matches:
                print(f"Exact matches for '{original_key}': {exact_key_matches}")
                if print_values:
                    for match in exact_key_matches:
                        print(f"{match} -> {self[match]}")
            else:
                if partial_matches:
                    print(
                        f"No exact matches found for '{original_key}'. Best matches are: {partial_matches}"
                    )
                else:
                    print(f"No matches found for '{original_key}'")

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
        time = round(time)
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
