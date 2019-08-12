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


    **Extends:** :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def _get_standard_fundamental_variables(self, T):
        param = self.param
        T_n, T_s, T_p = T.orphans

        T_av = pybamm.x_average(T)

        q = self._flux_law(T)

        variables = {
            "X-averaged negative electrode temperature": pybamm.x_average(T_n),
            "X-averaged negative electrode temperature [K]": param.Delta_T
            * pybamm.x_average(T_n)
            + param.T_ref,
            "Negative electrode temperature": T_n,
            "Negative electrode temperature [K]": param.Delta_T * T_n + param.T_ref,
            "X-averaged separator temperature": pybamm.x_average(T_s),
            "X-averaged separator temperature [K]": param.Delta_T
            * pybamm.x_average(T_s)
            + param.T_ref,
            "Separator temperature": T_s,
            "Separator temperature [K]": param.Delta_T * T_s + param.T_ref,
            "X-averaged positive electrode temperature": pybamm.x_average(T_p),
            "X-averaged positive electrode temperature [K]": param.Delta_T
            * pybamm.x_average(T_p)
            + param.T_ref,
            "Positive electrode temperature": T_p,
            "Positive electrode temperature [K]": param.Delta_T * T_p + param.T_ref,
            "Cell temperature": T,
            "Cell temperature [K]": param.Delta_T * T + param.T_ref,
            "X-averaged cell temperature": T_av,
            "X-averaged cell temperature [K]": param.Delta_T * T_av + param.T_ref,
            "Volume-averaged cell temperature": T_av,
            "Volume-averaged cell temperature [K]": param.Delta_T * T_av
            + param.T_ref,
            "Heat flux": q,
            "Heat flux [W.m-2]": q,
        }

        return variables

    def _get_standard_coupled_variables(self, variables):

        param = self.param

        T = variables["Cell temperature"]
        T_n, _, T_p = T.orphans

        j_n = variables["Negative electrode interfacial current density"]
        j_p = variables["Positive electrode interfacial current density"]

        eta_r_n = variables["Negative electrode reaction overpotential"]
        eta_r_p = variables["Positive electrode reaction overpotential"]

        dUdT_n = variables["Negative electrode entropic change"]
        dUdT_p = variables["Positive electrode entropic change"]

        i_e = variables["Electrolyte current density"]
        phi_e = variables["Electrolyte potential"]

        i_s_n = variables["Negative electrode current density"]
        i_s_p = variables["Positive electrode current density"]
        phi_s_n = variables["Negative electrode potential"]
        phi_s_p = variables["Positive electrode potential"]

        Q_ohm_s_n = -pybamm.inner(i_s_n, pybamm.grad(phi_s_n))
        Q_ohm_s_s = pybamm.FullBroadcast(0, ["separator"], "current collector")
        Q_ohm_s_p = -pybamm.inner(i_s_p, pybamm.grad(phi_s_p))
        Q_ohm_s = pybamm.Concatenation(Q_ohm_s_n, Q_ohm_s_s, Q_ohm_s_p)

        Q_ohm_e = -pybamm.inner(i_e, pybamm.grad(phi_e))

        Q_ohm = Q_ohm_s + Q_ohm_e

        Q_rxn_n = j_n * eta_r_n
        Q_rxn_p = j_p * eta_r_p
        Q_rxn = pybamm.Concatenation(
            *[
                Q_rxn_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rxn_p,
            ]
        )

        Q_rev_n = j_n * (param.Theta ** (-1) + T_n) * dUdT_n
        Q_rev_p = j_p * (param.Theta ** (-1) + T_p) * dUdT_p
        Q_rev = pybamm.Concatenation(
            *[
                Q_rev_n,
                pybamm.FullBroadcast(0, ["separator"], "current collector"),
                Q_rev_p,
            ]
        )

        Q = Q_ohm + Q_rxn + Q_rev
        Q_av = pybamm.x_average(Q)

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
                "X-averaged total heating": Q_av,
                "X-averaged total heating [A.V.m-3]": param.i_typ
                * param.potential_scale
                * Q_av
                / param.L_x,
                "Volume-averaged total heating": Q_av,
                "Volume-averaged total heating [A.V.m-3]": param.i_typ
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

    def set_initial_conditions(self, variables):

        T, _, _ = self._unpack(variables)

        self.initial_conditions = {T: self.param.T_init}
