import importlib.metadata
import sys
import textwrap
import warnings
from collections.abc import Mapping
from typing import Callable


class EntryPoint(Mapping):
    """
    Access via :py:data:`pybamm.parameter_sets` for parameter_sets
    Access via :py:data:`pybamm.Model` for Models

    Examples
    --------
    Listing available parameter sets:
        >>> import pybamm
        >>> list(pybamm.parameter_sets)  # doctest: +ELLIPSIS
        ['Ai2020', 'Chayambuka2022', ...]

    Listing available models:
        >>> import pybamm
        >>> list(pybamm.dispatch.models)
        ['SPM']

    Get the docstring for a parameter set or model:
        >>> print(pybamm.parameter_sets.get_docstring("Ai2020"))  # doctest: +ELLIPSIS
        <BLANKLINE>
        Parameters for the Enertech cell (Ai2020), from the papers :footcite:t:`Ai2019`...

        >>> print(pybamm.dispatch.models.get_docstring("SPM"))  # doctest: +ELLIPSIS
        <BLANKLINE>
        Single Particle Model (SPM) of a lithium-ion battery...
    """

    _instances = 0

    def __init__(self, group):
        """Dict of entry points for parameter sets or models, lazily load entry points as"""
        if not hasattr(
            self, "initialized"
        ):  # Ensure __init__ is called once per instance
            self.initialized = True
            EntryPoint._instances += 1
            self._all_entries = dict()
            self.group = group
            for entry_point in self.get_entries(self.group):
                self._all_entries[entry_point.name] = entry_point

    @staticmethod
    def get_entries(group_name):
        """Wrapper for the importlib version logic"""
        if sys.version_info < (3, 10):  # pragma: no cover
            return importlib.metadata.entry_points()[group_name]
        else:
            return importlib.metadata.entry_points(group=group_name)

    def __new__(cls, group):
        """Ensure only two instances of entry points exist, one for parameter sets and the other for models"""
        if EntryPoint._instances < 2:
            cls.instance = super().__new__(cls)
        return cls.instance

    def get_class(self, key) -> Callable:
        """Return the class without instantiating it"""
        return self._load_entry_point(key)

    def __getitem__(self, key) -> dict:
        return self._load_entry_point(key)()

    def _load_entry_point(self, key) -> Callable:
        """Check that ``key`` is a registered ``parameter_sets`` or ``models` ,
        and return the entry point for the parameter set/model, loading it needed."""
        if key not in self._all_entries:
            raise KeyError(f"Unknown parameter set or model: {key}")
        ps = self._all_entries[key]
        try:
            ps = self._all_entries[key] = ps.load()
        except AttributeError:
            pass
        return ps

    def __iter__(self):
        return self._all_entries.__iter__()

    def __len__(self) -> int:
        return len(self._all_entries)

    def get_docstring(self, key):
        """Return the docstring for the ``key`` parameter set or model"""
        return textwrap.dedent(self._load_entry_point(key).__doc__)

    def __getattribute__(self, name):
        try:
            # For backwards compatibility, parameter sets that used to be defined in
            # this file now return the name as a string, which will load the same
            # parameter set as before when passed to `ParameterValues`
            # Bypass the overloaded __getitem__ and __iter__ to avoid recursion
            _all_entries = super().__getattribute__("_all_entries")
            group = super().__getattribute__("group")

            # Only show deprecation warning for parameter sets, not models
            if name in _all_entries and group == "pybamm_parameter_sets":
                msg = (
                    f"Parameter sets should be called directly by their name ({name}), "
                    f"instead of via pybamm.parameter_sets (pybamm.parameter_sets.{name})."
                )
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return name
        except AttributeError:
            pass  # Handle the attribute error normally

        return super().__getattribute__(name)


#: Singleton Instance of :class:EntryPoint initialised with pybamm_parameter_sets"""
parameter_sets = EntryPoint(group="pybamm_parameter_sets")

#: Singleton Instance of :class:EntryPoint initialised with pybamm_models"""
models = EntryPoint(group="pybamm_models")


def Model(model: str, options=None, *args, **kwargs):  # doctest: +SKIP
    """
    Returns the loaded model object
    Parameters
    ----------
    model : str
        The model name or author name of the model mentioned at the model entry point.
    options : dict, optional
        Options to pass to the model constructor.
    *args
        Additional positional arguments to pass to the model constructor.
    **kwargs
        Additional keyword arguments to pass to the model constructor.
    Returns
    -------
    pybamm.model
        Model object of the initialised model.
    Examples
    --------
    Listing available models:
        >>> import pybamm
        >>> list(pybamm.dispatch.models)
        ['SPM']
        >>> pybamm.Model('SPM') # doctest: +SKIP
        <pybamm.models.full_battery_models.lithium_ion.spm.SPM object>
    """
    model_class = models.get_class(model)

    # Pass options as the first argument if provided, then *args and **kwargs
    if options is not None:
        return model_class(options, *args, **kwargs)
    else:
        return model_class(*args, **kwargs)
