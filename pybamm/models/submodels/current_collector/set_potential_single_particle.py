#
# Class for one-dimensional current collectors
#
import pybamm


class SetPotentialSingleParticle1plus1D(pybamm.BaseSubModel):
    """A submodel 1D current collectors which *doesn't* update the potentials
    during solve. This class uses the current-voltage relationship from the
    SPM(e) (see [1]_) to calculate the current.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    References
    ----------
    .. [1] SG Marquis, V Sulzer, R Timms, CP Please and SJ Chapman. “An asymptotic
           derivation of a single particle model with electrolyte”. In: arXiv preprint
           arXiv:1905.12553 (2019).


    **Extends:** :class:`pybamm.current_collector.BaseModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_potential_variables(self, phi_s_cn, phi_s_cp):
        """
        A private function to obtain the standard variables which
        can be derived from the potentials in the current collector.

        Parameters
        ----------
        phi_cc : :class:`pybamm.Symbol`
            The potential in the current collector.

        Returns
        -------
        variables : dict
            The variables which can be derived from the potential in the
            current collector.
        """
        param = self.param

        # Local potential difference
        V_cc = phi_s_cp - phi_s_cn

        phi_neg_tab = pybamm.BoundaryValue(phi_s_cn, "negative tab")
        phi_pos_tab = pybamm.BoundaryValue(phi_s_cp, "positive tab")

        variables = {
            "Negative current collector potential": phi_s_cn,
            "Negative current collector potential [V]": phi_s_cn
            * param.potential_scale,
            "Negative tab potential": phi_neg_tab,
            "Negative tab potential [V]": phi_neg_tab * param.potential_scale,
            "Positive tab potential": phi_pos_tab,
            "Positive tab potential [V]": param.U_p_ref
            - param.U_n_ref
            + phi_pos_tab * param.potential_scale,
            "Positive current collector potential": phi_s_cp,
            "Positive current collector potential [V]": param.U_p_ref
            - param.U_n_ref
            + phi_s_cp * param.potential_scale,
            "Local current collector potential difference": V_cc,
            "Local current collector potential difference [V]": param.U_p_ref
            - param.U_n_ref
            + V_cc * param.potential_scale,
        }

        return variables

    def _get_standard_current_variables(self, i_cc, i_boundary_cc):
        """
        A private function to obtain the standard variables which
        can be derived from the current in the current collector.
        Parameters
        ----------
        i_cc : :class:`pybamm.Symbol`
            The current in the current collector.
        i_boundary_cc : :class:`pybamm.Symbol`
            The current leaving the current collector and going into the cell
        Returns
        -------
        variables : dict
            The variables which can be derived from the current in the current
            collector.
        """

        # TO DO: implement grad in 2D to get i_cc
        # just need this to get 1D models working for now
        variables = {"Current collector current density": i_boundary_cc}

        return variables

    def get_fundamental_variables(self):

        phi_s_cn = pybamm.standard_variables.phi_s_cn
        phi_s_cp = pybamm.standard_variables.phi_s_cp

        variables = self._get_standard_potential_variables(phi_s_cn, phi_s_cp)

        # TO DO: grad not implemented for 2D yet
        i_cc = pybamm.Scalar(0)
        i_boundary_cc = pybamm.standard_variables.i_boundary_cc

        variables.update(self._get_standard_current_variables(i_cc, i_boundary_cc))

        return variables

    def set_rhs(self, variables):
        phi_s_cn = variables["Negative current collector potential"]
        phi_s_cp = variables["Positive current collector potential"]

        # Dummy equations so that PyBaMM doesn't change the potentials during solve
        # i.e. d_phi/d_t = 0. Potentials are set externally between steps.
        self.rhs = {phi_s_cn: pybamm.Scalar(0), phi_s_cp: pybamm.Scalar(0)}

    def set_algebraic(self, variables):
        ocp_p_av = variables["X-averaged positive electrode open circuit potential"]
        ocp_n_av = variables["X-averaged negative electrode open circuit potential"]
        eta_r_n_av = variables["X-averaged negative electrode reaction overpotential"]
        eta_r_p_av = variables["X-averaged positive electrode reaction overpotential"]
        eta_e_av = variables["X-averaged electrolyte overpotential"]
        delta_phi_s_n_av = variables["X-averaged negative electrode ohmic losses"]
        delta_phi_s_p_av = variables["X-averaged positive electrode ohmic losses"]

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
