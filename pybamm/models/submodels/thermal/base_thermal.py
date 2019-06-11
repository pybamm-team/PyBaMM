#
# Base class for thermal effects
#
import pybamm


class BaseThermal(pybamm.BaseSubModel):
    """Base class for thermal effects

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def __init__(self, param):
        super().__init__(param)

    def get_standard_derived_variables(self, variables):

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

        Q_rev_n = j_n * (self.param.Theta ** (-1) + T_n) * dUdT_n
        Q_rev_p = j_p * (self.param.Theta ** (-1) + T_p) * dUdT_p

        Q_rev = pybamm.Concatenation(
            *[Q_rev_n, pybamm.Broadcast(0, ["separator"]), Q_rev_p]
        )

        Q = Q_ohm + Q_rxn + Q_rev

        variables.update({})

        variables.update(self.get_dimensional_variables(variables))
        return variables

    def _flux_law(self, c):
        raise NotImplementedError

    def _unpack(self, variables):
        raise NotImplementedError

    def get_dimensional_variables(self, variables):

        c_s = variables[self._domain + " particle concentration"]
        c_s_xav = variables[
            "X-average " + self._domain.lower() + " particle concentration"
        ]
        c_s_surf = variables[self._domain + " particle surface concentration"]
        c_s_surf_av = variables[
            "Average " + self._domain.lower() + " particle surface concentration"
        ]

        if self._domain == "Negative":
            c_scale = self.param.c_n_max
        elif self._domain == "Positive":
            c_scale = self.param.c_p_max

        variables.update(
            {
                self._domain + " particle concentration [mol.m-3]": c_scale * c_s,
                "X-average "
                + self._domain.lower()
                + " particle concentration [mol.m-3]": c_scale * c_s_xav,
                self._domain
                + " particle surface concentration [mol.m-3]": c_scale * c_s_surf,
                "Average "
                + self._domain.lower()
                + " particle surface concentration [mol.m-3]": c_scale * c_s_surf_av,
            }
        )

        return variables

    def _initial_conditions(self, variables):

        c, _, _ = self._unpack(variables)

        if self._domain == "Negative":
            c_init = self.param.c_n_init

        elif self._domain == "Positive":
            c_init = self.param.c_p_init

        else:
            pybamm.DomainError

        initial_conditions = {c: c_init}

        return initial_conditions

