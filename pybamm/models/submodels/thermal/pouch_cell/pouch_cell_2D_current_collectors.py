#
# Class for two-dimensional thermal submodel for use in the "2+1D" pouch cell model
#
import pybamm

from .base_pouch_cell import BasePouchCell


class CurrentCollector2D(BasePouchCell):
    """Class for 2D thermal model for use in 2+1D pouch cell models

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    **Extends:** :class:`pybamm.thermal.pouch_cell.BasePouchCell`
    """

    def __init__(self, param):
        super().__init__(param, cc_dimension=2)

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

        # Add boundary source term which accounts for surface cooling around
        # the edge of the domain in the weak formulation.
        # TODO: update to allow different cooling conditions at the tabs
        self.rhs = {
            T_av: (
                pybamm.laplacian(T_av)
                + self.param.B * pybamm.source(Q_av, T_av)
                + cooling_coeff * pybamm.source(T_av - T_amb, T_av)
                - (self.param.h / self.param.delta)
                * pybamm.source(T_av - T_amb, T_av, boundary=True)
            )
            / (self.param.C_th * self.param.rho)
        }

    def set_boundary_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        # Dummy no flux boundary conditions since cooling at the the tabs is
        # accounted for in the boundary source term in the weak form of the
        # governing equation
        # TODO: update to allow different cooling conditions at the tabs
        self.boundary_conditions = {
            T_av: {
                "negative tab": (pybamm.Scalar(0), "Neumann"),
                "positive tab": (pybamm.Scalar(0), "Neumann"),
            }
        }

    def set_initial_conditions(self, variables):
        T_av = variables["X-averaged cell temperature"]
        self.initial_conditions = {T_av: self.param.T_init}
