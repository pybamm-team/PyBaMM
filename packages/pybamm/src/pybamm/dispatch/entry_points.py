import importlib.metadata
import textwrap
from collections.abc import Callable, Mapping


class EntryPoint(Mapping):
    """
    A mapping interface for accessing PyBaMM models and parameter sets through entry points.

    This class provides a unified way to load, and instantiate PyBaMM models
    and parameter sets that have been registered as entry points.

    Access via :py:data:`pybamm.parameter_sets` for parameter sets - provides access
    to all registered battery parameter sets (e.g., 'Chen2020', 'Ai2020') that can
    be used to parameterise battery models.

    Access via :py:data:`pybamm.Model` for models - provides access to all registered
    battery models (e.g., 'SPM', 'DFN').

    .. attention::

        This feature is currently experimental.

    Examples
    --------
    Listing available parameter sets:
        >>> import pybamm
        >>> list(pybamm.parameter_sets)  # doctest: +ELLIPSIS
        ['Ai2020', 'Chayambuka2022', ...]

    Listing available models:
        >>> import pybamm
        >>> list(pybamm.dispatch.models) # doctest: +ELLIPSIS
        ['DFN', 'MPM', ...]

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
        """Dict of entry points for parameter sets or models to lazily load as entry points"""
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
        return importlib.metadata.entry_points(group=group_name)

    def __new__(cls, group):
        """Ensure only two instances of entry points exist, one for parameter sets and the other for models"""
        if EntryPoint._instances < 2:
            cls.instance = super().__new__(cls)
        return cls.instance

    def _get_class(self, key) -> Callable:
        """Return the class without instantiating it"""
        return self._load_entry_point(key)

    def __getitem__(self, key) -> dict:
        return self._load_entry_point(key)()

    def _load_entry_point(self, key) -> Callable:
        """Check that ``key`` is a registered ``parameter_sets`` or ``models` ,
        and return the entry point for the parameter set/model, loading it needed."""
        if key not in self._all_entries:
            raise KeyError(f"Unknown parameter set or model: {key}")
        entry_point = self._all_entries[key]
        try:
            entry_point = self._all_entries[key] = entry_point.load()
        except AttributeError:
            # Parameter sets cannot be loaded so returning the default entry_point
            pass
        return entry_point

    def __iter__(self):
        return self._all_entries.__iter__()

    def __len__(self) -> int:
        return len(self._all_entries)

    def get_docstring(self, key):
        """Return the docstring for the ``key`` parameter set or model"""
        return textwrap.dedent(self._load_entry_point(key).__doc__)

    def __getattribute__(self, name):
        return super().__getattribute__(name)


#: Singleton Instance of :class:EntryPoint initialised with pybamm_parameter_sets"""
parameter_sets = EntryPoint(group="pybamm_parameter_sets")

#: Singleton Instance of :class:EntryPoint initialised with pybamm_models"""
models = EntryPoint(group="pybamm_models")


def Model(model: str, *args, **kwargs):
    """
    Returns the loaded model object
    Note: This feature is in its experimental phase.

    Parameters
    ----------
    model : str
        The model name or author name of the model mentioned at the model entry point.
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
        >>> list(pybamm.dispatch.models) # doctest: +ELLIPSIS
        ['DFN', 'MPM', ...]
        >>> pybamm.Model('SPM') # doctest: +SKIP
        <pybamm.models.full_battery_models.lithium_ion.spm.SPM object>
    """
    model_class = models._get_class(model)

    return model_class(*args, **kwargs)
