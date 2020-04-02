#
# Class for one-dimensional (x-direction) thermal submodel
#
import pybamm

from .base_thermal import BaseThermal


class OneDimensionalX(BaseThermal):
    """Class for one-dimensional (x-direction) thermal submodel.
    Note: this model assumes infinitely large electrical and thermal conductivity
    in the current collectors, so that the contribution to the Ohmic heating
    from the current collectors is zero and the boundary conditions are applied
    at the edges of the electrodes (at x=0 and x=1, in non-dimensional coordinates).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.BaseThermal`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):
        T_n = pybamm.standard_variables.T_n
        T_s = pybamm.standard_variables.T_s
        T_p = pybamm.standard_variables.T_p
        T_cn = pybamm.BoundaryValue(T_n, "left")
        T_cp = pybamm.BoundaryValue(T_p, "right")

        T = pybamm.Concatenation(T_n, T_s, T_p)
        T_x_av = self._x_average(T, T_cn, T_cp)
        T_vol_av = self._yz_average(T_x_av)

        variables = self._get_standard_fundamental_variables(
            T_cn, T_n, T_s, T_p, T_cp, T_x_av, T_vol_av
        )
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_coupled_variables(variables))
        return variables

    def set_rhs(self, variables):
        T = variables["Cell temperature"]
        Q = variables["Total heating"]

        # Fourier's law for heat flux
        q = -self.param.lambda_k * pybamm.grad(T)

        self.rhs = {
            T: (-pybamm.div(q) / self.param.delta ** 2 + self.param.B * Q)
            / (self.param.C_th * self.param.rho_k)
        }

    def set_boundary_conditions(self, variables):
        T = variables["Cell temperature"]
        T_n_left = pybamm.boundary_value(T, "left")
        T_p_right = pybamm.boundary_value(T, "right")
        T_amb = variables["Ambient temperature"]

        self.boundary_conditions = {
            T: {
                "left": (
                    self.param.h * (T_n_left - T_amb) / self.param.lambda_n,
                    "Neumann",
                ),
                "right": (
                    -self.param.h * (T_p_right - T_amb) / self.param.lambda_p,
                    "Neumann",
                ),
            }
        }

    def set_initial_conditions(self, variables):
        T = variables["Cell temperature"]
        self.initial_conditions = {T: self.param.T_init}

    def _current_collector_heating(self, variables):
        """
        In the limit of infinitely large current collector conductivity, the
        Ohmic heating in the current collectors in zero
        """
        Q_s_cn = pybamm.Scalar(0)
        Q_s_cp = pybamm.Scalar(0)
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """
        Computes the y-z average by integration over y and z
        In the 1D case this is just equal to the input variable
        """
        return var
