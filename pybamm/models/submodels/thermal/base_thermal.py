#
# Base class for thermal effects
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_fundamental_variables(self, T, T_av):
        param = self.param
        T_n, T_s, T_p = T.orphans

        q = self._flux_law(T)

        variables = {
            "Negative electrode temperature": T_n,
            "Negative electrode temperature [K]": param.Delta_T * T_n + param.T_ref,
            "Separator temperature": T_s,
            "Separator temperature [K]": param.Delta_T * T_s + param.T_ref,
            "Positive electrode temperature": T_p,
            "Positive electrode temperature [K]": param.Delta_T * T_p + param.T_ref,
            "Cell temperature": T,
            "Cell temperature [K]": param.Delta_T * T + param.T_ref,
            "Average cell temperature": T_av,
            "Average cell temperature [K]": param.Delta_T * T_av + param.T_ref,
            "Heat flux": q,
            "Heat flux [W.m-2]": q,
        }

        return variables

    def _get_standard_coupled_variables(self, variables):

        param = self.param

        T = variables["Cell temperature"]
        T_n, _, T_p = T.orphans

        j_n = variables["Negative electode interfacial current density"]
        j_p = variables["Positive electode interfacial current density"]

        eta_r_n = variables["Negative reaction overpotential"]
        eta_r_p = variables["Positive reaction overpotential"]

        dUdT_n = variables["Negative entropic change"]
        dUdT_p = variables["Positive entropic change"]

        i_e = variables["Electrolyte current density"]
        phi_e = variables["Electrolyte potential"]

        # phi_s_n = variables["Negative electrode potential"]
        # phi_s_s = pybamm.Broadcast(0, ["separator"])
        # phi_s_p = variables["Positive electrode potential"]
        # phi_s = pybamm.Concatenation(phi_s_n, phi_s_s, phi_s_p)

        # TODO: add ohmic heating from solid
        # Q_ohm = -i_s * pybamm.grad(phi_s) - i_e * pybamm.grad(phi_e)
        Q_ohm = -pybamm.inner(i_e, pybamm.grad(phi_e))

        Q_rxn_n = j_n * eta_r_n
        Q_rxn_p = j_p * eta_r_p
        Q_rxn = pybamm.Concatenation(
            *[Q_rxn_n, pybamm.Broadcast(0, ["separator"]), Q_rxn_p]
        )

        Q_rev_n = j_n * (param.Theta ** (-1) + T_n) * dUdT_n
        Q_rev_p = j_p * (param.Theta ** (-1) + T_p) * dUdT_p
        Q_rev = pybamm.Concatenation(
            *[Q_rev_n, pybamm.Broadcast(0, ["separator"]), Q_rev_p]
        )

        Q = Q_ohm + Q_rxn + Q_rev
        Q_av = pybamm.average(Q)

        variables.update(
            {
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
                "Total heating": Q,
                "Total heating [A.V.m-3]": param.i_typ
                * param.potential_scale
                * Q
                / param.L_x,
                "Average total heating": Q_av,
                "Average total heating [A.V.m-3]": param.i_typ
                * param.potential_scale
                * Q_av
                / param.L_x,
            }
        )

        # TODO: add units for heat flux

        return variables

    def _flux_law(self, T):
        raise NotImplementedError

    def _unpack(self, variables):
        raise NotImplementedError

    def _initial_conditions(self, variables):

        T = self._unpack(variables)

        initial_conditions = {T: self.param.T_init}

        return initial_conditions

