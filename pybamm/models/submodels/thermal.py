#
# Equation classes for thermal effects
#
import pybamm


class Thermal(pybamm.BaseSubModel):
    """Thermal effects by conservation of energy across the whole cell.

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def unpack(self, variables, reactions):
        """
        Unpack a dictionary of variables.

        Parameters
        ----------
        variables : dict
            Dictionary of variables to unpack
        domain : list of str
            Domain in which to unpack the variables
        """

        param = self.set_of_parameters

        T = variables.get("Cell temperature")
        T_n, _, T_p = T.orphans
        T_av = variables.get("Average cell temperature")
        # can maybe split this out to Internal cell temperature,
        # Negative current collector cell temperature and
        # Positive current collector cell temperature
        # in the 2D case

        # i_s_n = variables.get("Negative electrode current density")
        # i_s_s = pybamm.Broadcast(0, ["separator"])
        # i_s_p = variables.get("Positive electrode current density")
        # i_s = pybamm.Concatenation(i_s_n, i_s_s, i_s_p)

        i_e = variables.get("Electrolyte current density")

        # phi_s_n = variables.get("Negative electrode potential")
        # phi_s_s = pybamm.Broadcast(0, ["separator"])
        # phi_s_p = variables.get("Positive electrode potential")
        # phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        phi_e = variables.get("Electrolyte potential")

        j_n = reactions["main"]["neg"]["aj"]
        j_p = reactions["main"]["pos"]["aj"]

        eta_r_n = variables.get("Negative reaction overpotential")
        eta_r_p = variables.get("Positive reaction overpotential")

        # TODO: add ohmic heating from solid
        # Q_ohm = -i_s * pybamm.grad(phi_s) - i_e * pybamm.grad(phi_e)
        Q_ohm = -pybamm.inner(i_e, pybamm.grad(phi_e))

        Q_rxn_n = j_n * eta_r_n
        Q_rxn_p = j_p * eta_r_p

        Q_rxn = pybamm.Concatenation(
            *[Q_rxn_n, pybamm.Broadcast(0, ["separator"]), Q_rxn_p]
        )

        c_s_n_surf = variables.get("Negative particle surface concentration")
        c_s_p_surf = variables.get("Positive particle surface concentration")

        Q_rev_n = j_n * (param.Theta ** (-1) + T_n) * param.dUdT_n(c_s_n_surf)
        Q_rev_p = j_p * (param.Theta ** (-1) + T_p) * param.dUdT_p(c_s_p_surf)

        Q_rev = pybamm.Concatenation(
            *[Q_rev_n, pybamm.Broadcast(0, ["separator"]), Q_rev_p]
        )

        return T, T_av, Q_ohm, Q_rxn, Q_rev

    def unpack_post(self, variables):
        """ Unpack variables for post-processing """
        i_boundary_cc = variables["Current collector current density"]
        ocp_p = variables["Positive electrode open circuit potential"]
        eta_r_p = variables["Positive reaction overpotential"]
        phi_e = variables["Electrolyte potential"]

        ocp_p_av = pybamm.average(ocp_p)
        eta_r_p_av = pybamm.average(eta_r_p)
        phi_e_p = phi_e.orphans[2]
        phi_e_p_av = pybamm.average(phi_e_p)

        return i_boundary_cc, ocp_p_av, eta_r_p_av, phi_e_p_av

    def set_full_differential_system(self, variables, reactions):

        param = self.set_of_parameters
        T, _, Q_ohm, Q_rxn, Q_rev = self.unpack(variables, reactions)
        q = -param.lambda_k * pybamm.grad(T)

        self.rhs = {
            T: (-pybamm.div(q) + param.delta ** 2 * param.B * (Q_ohm + Q_rxn + Q_rev))
            / (param.delta ** 2 * param.C_th * param.rho_k)
        }
        self.algebraic = {}

        T_n_left = pybamm.boundary_value(T, "left")
        T_p_right = pybamm.boundary_value(T, "right")
        self.boundary_conditions = {
            T: {
                "left": (-param.h * T_n_left / param.lambda_k, "Neumann"),
                "right": (-param.h * T_p_right / param.lambda_k, "Neumann"),
            }
        }
        self.initial_conditions = {T: param.T_init}
        self.variables = self.get_variables(T, q, Q_ohm, Q_rxn, Q_rev)

    def set_x_lumped_differential_system(self, variables, reactions):

        param = self.set_of_parameters
        _, T_av, Q_ohm, Q_rxn, Q_rev = self.unpack(variables, reactions)

        Q = Q_ohm + Q_rxn + Q_rev
        Q_av = pybamm.average(Q)
        self.rhs = {
            T_av: (param.B * Q_av - 2 * param.h / (param.delta ** 2) * T_av)
            / (param.C_th * param.rho)
        }
        self.algebraic = {}
        self.boundary_conditions = {}
        self.initial_conditions = {T_av: param.T_init}

        T_n = pybamm.Broadcast(T_av, ["negative electrode"])
        T_s = pybamm.Broadcast(T_av, ["separator"])
        T_p = pybamm.Broadcast(T_av, ["positive electrode"])

        T = pybamm.Concatenation(T_n, T_s, T_p)
        q = pybamm.Broadcast(
            pybamm.Scalar(0), ["negative electrode", "separator", "positive electrode"]
        )

        self.variables = self.get_variables(T, q, Q_ohm, Q_rxn, Q_rev)

    def get_variables(self, T_k, q, Q_ohm, Q_rxn, Q_rev):

        param = self.set_of_parameters

        T_n, T_s, T_p = T_k.orphans
        T_k_av = pybamm.average(T_k)

        return {
            "Negative electrode temperature": T_n,
            "Negative electrode temperature [K]": param.Delta_T * T_n + param.T_ref,
            "Separator temperature": T_s,
            "Separator temperature [K]": param.Delta_T * T_s + param.T_ref,
            "Positive electrode temperature": T_p,
            "Positive electrode temperature [K]": param.Delta_T * T_p + param.T_ref,
            "Cell temperature": T_k,
            "Cell temperature [K]": param.Delta_T * T_k + param.T_ref,
            "Average cell temperature": T_k_av,
            "Average cell temperature [K]": param.Delta_T * T_k_av + param.T_ref,
            "Heat flux": q,
            "Heat flux [W.m-2]": q,
            "Ohmic heating": Q_ohm,
            "Ohmic heating [A.V.m-3]": param.i_typ
            * param.potential_scale
            * Q_ohm
            / param.L_x,
            "Irreversible electrochemical heating": Q_rxn,
            "Irreversible electrochemical heating [A.V.m-3]": param.i_typ
            * param.potential_scale
            * Q_rxn
            / param.L_x,
            "Reversible heating": Q_rev,
            "Reversible heating [A.V.m-3]": param.i_typ
            * param.potential_scale
            * Q_rev
            / param.L_x,
        }

