#
# Class for two-dimensional current collectors - Single-Particle formulation
#
from .potential_pair import PotentialPair2plus1D


class SingleParticlePotentialPair(PotentialPair2plus1D):
    """A submodel for Ohm's law plus conservation of current in the current collectors,
    which uses the voltage-current relationship from the SPM(e).

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.current_collector.PotentialPair2plus1D`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_coupled_variables(self, variables):
        ocp_p_av = variables["X-averaged positive electrode open circuit potential"]
        ocp_n_av = variables["X-averaged negative electrode open circuit potential"]
        eta_r_n_av = variables["X-averaged negative electrode reaction overpotential"]
        eta_r_p_av = variables["X-averaged positive electrode reaction overpotential"]
        eta_e_av = variables["X-averaged electrolyte overpotential"]
        delta_phi_s_n_av = variables["X-averaged negative electrode ohmic losses"]
        delta_phi_s_p_av = variables["X-averaged positive electrode ohmic losses"]

        phi_s_cn = variables["Negative current collector potential"]

        local_voltage_expression = (
            ocp_p_av
            - ocp_n_av
            + eta_r_p_av
            - eta_r_n_av
            + eta_e_av
            + delta_phi_s_p_av
            - delta_phi_s_n_av
        )
        phi_s_cp = phi_s_cn + local_voltage_expression
        variables = self._get_standard_potential_variables(phi_s_cn, phi_s_cp)
        return variables
