#
# Class for two-dimensional current collectors
#
import pybamm
from .base_current_collector import BaseModel


class PotentialPair(BaseModel):
    """A submodel for Ohm's law plus conservation of current in the current collectors,
    which uses the voltage-current relationship from the SPM(e).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_fundamental_variables(self):

        phi_s_cn = pybamm.standard_variables.phi_s_cn

        variables = self._get_standard_negative_potential_variables(phi_s_cn)

        # TODO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.standard_variables.i_boundary_cc

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        return variables

    def set_algebraic(self, variables):

        param = self.param

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.algebraic = {
            phi_s_cn: pybamm.laplacian(phi_s_cn)
            - (param.sigma_cn * param.delta ** 2 / param.l_cn)
            * pybamm.source(i_boundary_cc, phi_s_cn),
            i_boundary_cc: pybamm.laplacian(phi_s_cp)
            + (param.sigma_cp * param.delta ** 2 / param.l_cp)
            * pybamm.source(i_boundary_cc, phi_s_cp),
        }

    def set_boundary_conditions(self, variables):

        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        param = self.param
        applied_current = param.current_with_time

        pos_tab_bc = -applied_current / (
            param.sigma_cp * param.delta ** 2 * param.l_tab_p * param.l_cp
        )

        # Boundary condition needs to be on the variables that go into the Laplacian,
        # even though phi_s_cp isn't a pybamm.Variable object
        self.boundary_conditions = {
            phi_s_cn: {
                "left": (pybamm.Scalar(0), "Dirichlet"),
                "right": (pybamm.Scalar(0), "Neumann"),
            },
            phi_s_cp: {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (pos_tab_bc, "Neumann"),
            },
        }

    def set_initial_conditions(self, variables):

        param = self.param
        applied_current = param.current_with_time
        phi_s_cn = variables["Negative current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            i_boundary_cc: applied_current / param.l_y / param.l_z,
        }
