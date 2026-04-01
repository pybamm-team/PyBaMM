import importlib.metadata
import textwrap
from collections.abc import Callable, Mapping
from pathlib import Path

import pooch
from platformdirs import user_cache_dir

from pybamm.expression_tree.operations.serialise import Serialise

APP_NAME = "pybamm"

MODELS_CACHE_DIR: Path = Path(user_cache_dir(APP_NAME)) / "models"


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


def _get_cache_dir() -> Path:
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_CACHE_DIR


def get_cache_path(url: str) -> Path:
    import hashlib

    cache_dir = _get_cache_dir()
    file_hash = hashlib.sha256(url.encode()).hexdigest()
    return cache_dir / f"{file_hash}.json"


def clear_model_cache() -> None:
    cache_dir = _get_cache_dir()
    for file in cache_dir.glob("*.json"):
        try:
            file.unlink()
        except OSError as e:
            print(f"Could not delete {file}: {e}")


def Model(
    model: str | None = None,
    url: str | None = None,
    force_download: bool = False,
    *args,
    **kwargs,
) -> object:
    """
    Returns the loaded model object.

    .. note::
        This feature is in its experimental phase.

    Parameters
    ----------
    model : str, optional
        The model name registered as a ``pybamm_models`` entry point
        (e.g. ``'SPM'``, ``'DFN'``).  Mutually exclusive with *url*.
    url : str, optional
        A direct URL pointing to a serialised PyBaMM model JSON file.
        The file is downloaded via :mod:`pooch` and cached locally under
        :data:`MODELS_CACHE_DIR`.  Mutually exclusive with *model*.
    force_download : bool, optional
        When *True*, re-download the model even if a local cached copy
        already exists.  Defaults to ``False``.
    *args
        Additional positional arguments forwarded to the model constructor
        (only meaningful when *model* is supplied).
    **kwargs
        Additional keyword arguments forwarded to the model constructor
        (only meaningful when *model* is supplied).

    Returns
    -------
    object
        An instantiated PyBaMM model object.

    Raises
    ------
    ValueError
        If neither or both of *model* and *url* are provided, or if the
        named *model* cannot be loaded.
    RuntimeError
        If the download from *url* fails.

    Examples
    --------
    Listing available models:
        >>> import pybamm
        >>> list(pybamm.dispatch.models) # doctest: +ELLIPSIS
        ['DFN', 'MPM', ...]
        >>> pybamm.Model('SPM') # doctest: +SKIP
        <pybamm.models.full_battery_models.lithium_ion.spm.SPM object>
    """
    if (model is None and url is None) or (model and url):
        raise ValueError("You must provide exactly one of `model` or `url`.")

    if url is not None:
        cache_dir = _get_cache_dir()
        import hashlib

        file_hash = hashlib.sha256(url.encode()).hexdigest()
        fname = f"{file_hash}.json"

        cached_path = cache_dir / fname
        if force_download and cached_path.exists():
            cached_path.unlink()

        already_cached = cached_path.exists()

        try:
            fetched = pooch.retrieve(
                url=url,
                known_hash=None,
                fname=fname,
                path=cache_dir,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model from URL: {e}") from e

        if already_cached:
            print(f"Using cached model at: {fetched}")
        else:
            print(f"Model cached at: {fetched}")

        return Serialise.load_custom_model(str(fetched))
    try:
        model_class = models._get_class(model)
        return model_class(*args, **kwargs)
    except Exception as e:
        raise ValueError(f"Could not load model '{model}': {e}") from e
