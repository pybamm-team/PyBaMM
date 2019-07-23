#
# Class for one-dimensional current collectors
#
import pybamm
from .base_current_collector import BaseModel


class OneDimensionalCurrentCollector(BaseModel):
    """A submodel 1D current collectors.

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
        phi_s_cp = pybamm.standard_variables.phi_s_cp

        variables = self._get_standard_potential_variables(phi_s_cn, phi_s_cp)

        # TO DO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.standard_variables.i_boundary_cc

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        return variables

    def set_algebraic(self, variables):

        ocp_p_av = variables["Average positive electrode open circuit potential"]
        ocp_n_av = variables["Average negative electrode open circuit potential"]
        eta_r_n_av = variables["Average negative electrode reaction overpotential"]
        eta_r_p_av = variables["Average positive electrode reaction overpotential"]
        eta_e_av = variables["Average electrolyte overpotential"]
        delta_phi_s_n_av = variables["Average negative electrode ohmic losses"]
        delta_phi_s_p_av = variables["Average positive electrode ohmic losses"]

        i_boundary_cc = variables["Current collector current density"]
        v_boundary_cc = variables["Local current collector potential difference"]

        # The voltage-current expression from the SPM(e)
        local_voltage_expression = (
            ocp_p_av
            - ocp_n_av
            + eta_r_p_av
            - eta_r_n_av
            + eta_e_av
            + delta_phi_s_p_av
            - delta_phi_s_n_av
        )

        self.algebraic = {i_boundary_cc: v_boundary_cc - local_voltage_expression}

    def set_rhs(self, variables):
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        self.rhs = {phi_s_cn: 0, phi_s_cp: 0}

    def set_initial_conditions(self, variables):

        param = self.param
        applied_current = param.current_with_time
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]
        i_boundary_cc = variables["Current collector current density"]

        self.initial_conditions = {
            phi_s_cn: pybamm.Scalar(0),
            phi_s_cp: param.U_p(param.c_p_init, param.T_ref)
            - param.U_n(param.c_n_init, param.T_ref),
            i_boundary_cc: applied_current / param.l_y / param.l_z,
        }
