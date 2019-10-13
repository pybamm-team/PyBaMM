#
# Interpolating class
#
import pybamm
import numpy as np
from scipy import interpolate


class Interpolant(pybamm.Function):
    """
    Interpolate data in 1D.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
    child : :class:`pybamm.Symbol`
    name : str, optional
    interpolator : str, optional
        Which interpolator to use ("linear", "pchip" or "cubic spline"). Default is
        "pchip".
    extrapolate : bool, optional
        Whether to extrapolate for points that are outside of the parametrisation
        range, or return NaN (following default behaviour from scipy). Default is True.
    """

    def __init__(self, data, child, name=None, interpolator="pchip", extrapolate=True):
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(
                "data should have exactly two columns (x and y) but has shape {}".format(
                    data.shape
                )
            )
        if interpolator == "linear":
            if extrapolate is True:
                fill_value = "extrapolate"
            else:
                fill_value = np.nan
            interpolating_function = interpolate.interp1d(
                data[:, 0], data[:, 1], fill_value=fill_value
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
        super().__init__(interpolating_function, child)
        # Overwrite name if given
        if name is not None:
            self.name = "interpolating function ({})".format(name)
        # Store information as attributes
        self.interpolator = interpolator
        self.extrapolate = extrapolate

    def _diff(self, variable):
        """
        Overwrite the base Function `_diff` to use `.derivative` directly instead of
        autograd.
        See :meth:`pybamm.Function._diff()`.
        """
        interpolating_function = self.function
        return pybamm.Function(interpolating_function.derivative(), *self.children)
