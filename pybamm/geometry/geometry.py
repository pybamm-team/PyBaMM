#
# Geometry class for storing the geometry of the model
#
import pybamm
import numbers


class Geometry(dict):
    """
    A geometry class to store the details features of the cell geometry.

    The values assigned to each domain are dictionaries containing the spatial variables
    in that domain, along with expression trees giving their min and maximum extents.
    For example, the following dictionary structure would represent a Geometry with a
    single domain "negative electrode", defined using the variable `x_n` which has a
    range from 0 to the pre-defined parameter `l_n`.

    .. code-block:: python

       {"negative electrode": {x_n: {"min": pybamm.Scalar(0), "max": l_n}}}

    **Extends**: :class:`dict`

    Parameters
    ----------

    geometries: dict
        The dictionary to create the geometry with

    """

    def __init__(self, geometry):
        super().__init__(**geometry)
        self._parameters = None

    @property
    def parameters(self):
        "Returns all the parameters in the geometry"
        if self._parameters is None:
            self._parameters = self._find_parameters()
        return self._parameters

    def _find_parameters(self):
        "Find all the parameters in the model"
        unpacker = pybamm.SymbolUnpacker((pybamm.Parameter, pybamm.InputParameter))

        def NestedDictValues(d):
            "Get all the values from a nested dict"
            for v in d.values():
                if isinstance(v, dict):
                    yield from NestedDictValues(v)
                else:
                    if isinstance(v, numbers.Number):
                        yield pybamm.Scalar(v)
                    else:
                        yield v

        all_parameters = unpacker.unpack_list_of_symbols(list(NestedDictValues(self)))
        return list(all_parameters.values())
