import warnings
import pkg_resources
import textwrap
from collections.abc import Mapping


class ParameterSets(Mapping):
    """
    Dict-like interface for accessing registered pybamm parameter sets.
    Access via :py:data:`pybamm.parameter_sets`

    Examples
    --------
    Listing available parameter sets:

    .. doctest::

        >>> import pybamm
        >>> list(pybamm.parameter_sets)
        ['Ai2020', 'Chen2020', ...]

    Get the docstring for a parameter set:

    .. doctest::

        >>> import pybamm
        >>> print(pybamm.parameter_sets.get_docstring("Ai2020"))
        <BLANKLINE>
        Parameters for the Enertech cell (Ai2020), from the papers:
        ...

    See also: :ref:`adding-parameter-sets`

    """

    def __init__(self):
        # Load Parameter Sets registered to `pybamm_parameter_set`
        ps = dict()
        for entry_point in pkg_resources.iter_entry_points("pybamm_parameter_set"):
            ps[entry_point.name] = entry_point.load()

        self.__all_parameter_sets = ps

    def __new__(cls):
        """Ensure only one instance of ParameterSets exists"""
        if not hasattr(cls, "instance"):
            cls.instance = super(ParameterSets, cls).__new__(cls)
        return cls.instance

    def __getitem__(self, key) -> dict:
        return self.__all_parameter_sets[key]()

    def __iter__(self):
        return self.__all_parameter_sets.__iter__()

    def __len__(self) -> int:
        return len(self.__all_parameter_sets)

    def get_docstring(self, key):
        """Return the docstring for the ``key`` parameter set"""
        return textwrap.dedent(self.__all_parameter_sets[key].__doc__)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError as error:
            # For backwards compatibility, parameter sets that used to be defined in
            # this file now return the name as a string, which will load the same
            # parameter set as before when passed to `ParameterValues`
            if name in self.__all_parameter_sets:
                out = name
            else:
                raise error
            warnings.warn(
                f"Parameter sets should be called directly by their name ({name}), "
                f"instead of via pybamm.parameter_sets (pybamm.parameter_sets.{name}).",
                DeprecationWarning,
            )
            return out


#: Singleton Instance of :class:ParameterSets """
parameter_sets = ParameterSets()
