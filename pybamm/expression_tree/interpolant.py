#
# Interpolating class
#
import pybamm
from scipy import interpolate


class Interpolant(pybamm.Function):
    """
    Interpolate data in 1D.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Numpy array of data to use for interpolation. Must have exactly two columns (x
        and y data)
    child : :class:`pybamm.Symbol`
        Node to use when evaluating the interpolant
    name : str, optional
        Name of the interpolant. Default is None, in which case the name "interpolating
        function" is given.
    interpolator : str, optional
        Which interpolator to use ("pchip" or "cubic spline"). Note that whichever
        interpolator is used must be differentiable (for ``Interpolator._diff``).
        Default is "cubic spline". Note that "pchip" may give slow results.
    extrapolate : bool, optional
        Whether to extrapolate for points that are outside of the parametrisation
        range, or return NaN (following default behaviour from scipy). Default is True.

    **Extends**: :class:`pybamm.Function`
    """

    def __init__(
        self,
        data,
        child,
        name=None,
        interpolator="cubic spline",
        extrapolate=True,
        entries_string=None,
    ):
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                """
                data should have exactly two columns (x and y) but has shape {}
                """.format(
                    data.shape
                )
            )
        elif interpolator == "pchip":
            interpolating_function = interpolate.PchipInterpolator(
                data[:, 0], data[:, 1], extrapolate=extrapolate
            )
        elif interpolator == "cubic spline":
            interpolating_function = interpolate.CubicSpline(
                data[:, 0], data[:, 1], extrapolate=extrapolate
            )
        else:
            raise ValueError("interpolator '{}' not recognised".format(interpolator))
        # Set name
        if name is not None and not name.startswith("interpolating function"):
            name = "interpolating function ({})".format(name)
        else:
            name = "interpolating function"
        self.data = data
        self.entries_string = entries_string
        super().__init__(
            interpolating_function, child, name=name, derivative="derivative"
        )
        # Store information as attributes
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.interpolator = interpolator
        self.extrapolate = extrapolate

    @property
    def entries_string(self):
        return self._entries_string

    @entries_string.setter
    def entries_string(self, value):
        # We must include the entries in the hash, since different arrays can be
        # indistinguishable by class, name and domain alone
        # Slightly different syntax for sparse and non-sparse matrices
        if value is not None:
            self._entries_string = value
        else:
            entries = self.data
            self._entries_string = entries.tostring()

    def set_id(self):
        """ See :meth:`pybamm.Symbol.set_id()`. """
        self._id = hash(
            (self.__class__, self.name, self.entries_string) + tuple(self.domain)
        )

    def _function_new_copy(self, children):
        """ See :meth:`Function._function_new_copy()` """
        return pybamm.Interpolant(
            self.data,
            *children,
            name=self.name,
            interpolator=self.interpolator,
            extrapolate=self.extrapolate,
            entries_string=self.entries_string
        )
