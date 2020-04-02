#
# Class for one-dimensional thermal submodel for use in the "1+1D" pouch cell model
#
import pybamm

from .base_pouch_cell import BasePouchCell


class CurrentCollector1D(BasePouchCell):
    """Class for 1D thermal model for use in 1+1D pouch cell models

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.pouch_cell.BasePouchCell`
    """

    def __init__(self, param):
        super().__init__(param)

    def set_rhs(self, variables):
        T_av = variables["X-averaged cell temperature"]
        Q_av = variables["X-averaged total heating"]
        T_amb = variables["Ambient temperature"]

        # Account for surface area to volume ratio of pouch cell in cooling
        # coefficient. Note: the factor 1/delta^2 comes from the choice of
        # non-dimensionalisation
        A = self.param.l_y * self.param.l_z
        V = self.param.l * self.param.l_y * self.param.l_z
        cooling_coeff = -2 * self.param.h * A / V / (self.param.delta ** 2)

        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * Q_av
                + cooling_coeff * (T_av - T_amb)
            )
            / (self.param.C_th * self.param.rho)
        }

    def set_boundary_conditions(self, variables):
        T_amb = variables["Ambient temperature"]
        T_av = variables["X-averaged cell temperature"]
        T_av_left = pybamm.boundary_value(T_av, "negative tab")
        T_av_right = pybamm.boundary_value(T_av, "positive tab")

        # Three boundary conditions here to handle the cases of both tabs at
        # the same side (top or bottom), or one either side. For both tabs on the
        # same side, T_av_left and T_av_right are equal, and the boundary condition
        # "no tab" is used on the other side.
        self.boundary_conditions = {
            T_av: {
                "negative tab": (
                    self.param.h * (T_av_left - T_amb) / self.param.delta,
                    "Neumann",
                ),
                "positive tab": (
                    -self.param.h * (T_av_right - T_amb) / self.param.delta,
                    "Neumann",
                ),
                "no tab": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}

    def _current_collector_heating(self, variables):
        """Returns the heat source terms in the 1D current collector"""
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        Q_s_cn = self.param.sigma_cn_prime * pybamm.inner(
            pybamm.grad(phi_s_cn), pybamm.grad(phi_s_cn)
        )
        Q_s_cp = self.param.sigma_cp_prime * pybamm.inner(
            pybamm.grad(phi_s_cp), pybamm.grad(phi_s_cp)
        )
        return Q_s_cn, Q_s_cp

    def _yz_average(self, var):
        """Computes the y-z average by integration over z (no y-direction)"""
        return pybamm.z_average(var)
