import importlib.metadata
import textwrap
from collections.abc import Callable, Mapping
from importlib.util import find_spec

from packaging.requirements import Requirement
from packaging.version import Version

import pybamm


def _entry_point_is_available(entry_point: importlib.metadata.EntryPoint) -> bool:
    if (
        entry_point.group != "pybamm_parameter_sets"
        or entry_point.value
        != "pybamm.input.parameters.lithium_ion."
        "MolicelP45B_AboutEnergy2026:get_parameter_values"
    ):
        return True

    try:
        if find_spec("bpx") is None:
            return False
        bpx_version = Version(importlib.metadata.version("bpx"))
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return False

    distribution = entry_point.dist
    if distribution is None:
        return False
    return any(
        bpx_version in requirement.specifier
        for requirement in map(Requirement, distribution.requires or ())
        if requirement.name == "bpx"
        and (
            requirement.marker is None or requirement.marker.evaluate({"extra": "bpx"})
        )
    )


class EntryPoint(Mapping):
    """
    A mapping interface for accessing PyBaMM models and parameter sets through entry points.

    Access via :py:data:`pybamm.parameter_sets` for parameter sets - provides access
    to installed battery parameter sets (e.g., 'Chen2020', 'Ai2020'). Sets needing
    an extra are listed by ``get_entries("pybamm_parameter_sets")``.

    Access via :py:data:`pybamm.Model` for models - provides access to all registered
    battery models (e.g., 'SPM', 'DFN').

    Names are listed in sorted order, whichever package registers them.

    .. attention::

        This feature is currently experimental.

    Examples
    --------
    Checking that a parameter set is available:
        >>> import pybamm
        >>> "Ai2020" in pybamm.parameter_sets
        True

    Checking that a model is available:
        >>> import pybamm
        >>> "SPM" in pybamm.dispatch.models
        True

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
            self._all_entries = {}
            self.group = group
            for entry_point in self.get_entries(self.group):
                if _entry_point_is_available(entry_point):
                    self._all_entries[entry_point.name] = entry_point

    @staticmethod
    def get_entries(group_name):
        """Return the entry points registered to ``group_name``, sorted by name"""
        # Without sorting, the order depends on which installed package is found first
        return sorted(
            importlib.metadata.entry_points(group=group_name),
            key=lambda entry: entry.name,
        )

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

    def __contains__(self, key) -> bool:
        return key in self._all_entries

    def __iter__(self):
        return self._all_entries.__iter__()

    def items(self):
        """Yield ``(name, value)`` pairs, skipping entries whose optional
        dependencies are not installed."""
        for key in self:
            try:
                yield key, self[key]
            except ModuleNotFoundError as error:
                pybamm.logger.info(f"Skipping '{key}': {error}")

    def values(self):
        """Yield values, skipping entries whose optional dependencies are not
        installed."""
        yield from (value for _, value in self.items())

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
    Checking that a model is available:
        >>> import pybamm
        >>> "SPM" in pybamm.dispatch.models
        True
        >>> pybamm.Model('SPM') # doctest: +SKIP
        <pybamm.models.full_battery_models.lithium_ion.spm.SPM object>
    """
    model_class = models._get_class(model)

    return model_class(*args, **kwargs)
