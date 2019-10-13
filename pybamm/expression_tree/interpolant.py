#
# Interpolating class
#
import pybamm
from scipy import interpolate


class Interpolant(pybamm.Function):
    def __init__(self, data, child):
        interpolating_function = interpolate.CubicSpline(
            data[:, 0], data[:, 1], extrapolate=True
        )
        super().__init__(interpolating_function, child)

    def _diff(self, variable):
        """ See :meth:`pybamm.Function._diff()`. """
        return pybamm.Function(
            self._interpolating_function.derivative(), *self.children
        )
