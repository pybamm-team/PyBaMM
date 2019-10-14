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
        self, data, child, name=None, interpolator="cubic spline", extrapolate=True
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
        self.name = "interpolating function"
        if name is not None:
            self.name += " ({})".format(name)
        super().__init__(
            interpolating_function, child, name=name, derivative="derivative"
        )
        # Store information as attributes
        self.interpolator = interpolator
        self.extrapolate = extrapolate
