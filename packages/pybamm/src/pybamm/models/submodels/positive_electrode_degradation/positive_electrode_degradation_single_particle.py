#
# Class for a single particle with positive electrode degradation
#

import pybamm

from .base_positive_electrode_degradation import BasePositiveElectrodeDegradation


class PositiveElectrodeDegradationSingleParticle(BasePositiveElectrodeDegradation):
    """
    Class for positive electrode degradation in a single x-averaged particle

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model must be 'Positive'

    **Extends:** :class:`pybamm.positive_electrode_degradation.BasePositiveElectrodeDegradation`
    """

    def __init__(self, param, domain="Positive"):
        if domain != "Positive":
            raise pybamm.DomainError(
                "Value of domain must be 'Positive' for phase transition degradation"
            )
        super().__init__(param, domain)
        pybamm.citations.register("Ghosh2021")
        pybamm.citations.register("Zhuo2023")

    def get_fundamental_variables(self):
        # Concentration in particle core
        c_c_xav = pybamm.Variable(
            "X-averaged positive core stoichiometry",
            domains={"primary": "positive core", "secondary": "current collector"},
            bounds=(0, 1),
        )
        c_c = pybamm.SecondaryBroadcast(c_c_xav, ["positive electrode"])

        # Oxygen concentration in degraded passivation layer/shell
        c_o_xav = pybamm.Variable(
            "X-averaged positive shell oxygen stoichiometry",
            domains={
                "primary": "positive shell oxygen",
                "secondary": "current collector",
            },
        )
        c_o = pybamm.SecondaryBroadcast(c_o_xav, ["positive electrode"])

        # location of core-shell boundary
        s_xav = pybamm.Variable(
            "X-averaged moving phase boundary location [m]",
            domain="current collector",
            bounds=(0, self.param.p.prim.R_typ),  # s cannot exceed the particle radius
        )

        variables = self._get_standard_concentration_variables(
            c_c=c_c,
            c_o=c_o,
            s=pybamm.PrimaryBroadcast(s_xav, ["positive electrode"]),
            c_c_xav=c_c_xav,
            c_o_xav=c_o_xav,
            s_xav=s_xav,
        )

        return variables

    def get_coupled_variables(self, variables):
        T_xav = variables["X-averaged positive electrode temperature [K]"]

        c_o_cent_av = variables["X-averaged positive shell center oxygen stoichiometry"]
        c_c_surf_av = variables["X-averaged positive core surface stoichiometry"]

        # Get s_dot
        K_1 = self.k_1_dimensional(T_xav)  # Appendix B 1st Paragraph  [m.s-1]
        K_2 = (
            self.k_2_dimensional(T_xav) * self.c_o_core_dim
        )  # Appendix B 1st Paragraph  [m.s-1]

        s_dot = -(  # Zhuo et al 2023 (13), Ghosh et al 2021 [C.12]
            K_1 - (K_2 * c_o_cent_av)
        ) * pybamm.EqualHeaviside(c_c_surf_av, self.c_p_thrd)

        # Get c_c and c_o at boundary
        c_c_xav = variables["X-averaged positive core stoichiometry"]
        c_o_xav = variables["X-averaged positive shell oxygen stoichiometry"]
        s_xav = variables["X-averaged moving phase boundary location [m]"]

        T_xav_c = pybamm.PrimaryBroadcast(T_xav, ["positive core"])
        T_xav_o = pybamm.PrimaryBroadcast(T_xav, ["positive shell oxygen"])

        # Defined in base_positive_electrode_degradation

        D_c = pybamm.surf(self.D_c_dimensional(c_c_xav, T_xav_c))
        D_o = pybamm.boundary_value(self.D_o_dimensional(c_o_xav, T_xav_o), "left")

        c_c_N_av = variables["X-averaged positive core surface cell stoichiometry"]
        c_o_1_av = variables[
            "X-averaged positive shell center cell oxygen stoichiometry"
        ]
        dx_cp_av = variables["X-averaged positive core surface cell length"]
        dx_co_av = variables["X-averaged positive shell center cell length of oxygen"]

        j_xav = variables[
            "X-averaged positive electrode interfacial current density [A.m-2]"
        ]

        # The interface boundary value is calculated from the applied boundary
        # condition, not extrapolated afterwards.

        F = self.param.F
        R_p = self.param.p.prim.R_typ
        c_p_max = self.param.p.prim.c_max
        d_c = D_c / (s_xav * dx_cp_av)  # Intermediary variable

        J = (R_p / s_xav) ** 2 * (j_xav / F) / c_p_max
        c_c_b_xav = ((s_dot * self.c_s_trap) + (c_c_N_av * d_c) - J) / (
            s_dot + d_c
        )  # Zhuo et al EQ 10 & A.2

        c_c_b = pybamm.PrimaryBroadcast(c_c_b_xav, ["positive electrode"])

        # Boundary oxygen concentration from applied bc

        d_o = D_o / ((R_p - s_xav) * dx_co_av)  # Intermediary variable

        c_o_b_xav = ((c_o_1_av * d_o) - (s_dot * self.c_o_core)) / (
            d_o - s_dot
        )  # Zhuo et al EQ 10 & A.2

        c_o_b = pybamm.PrimaryBroadcast(c_o_b_xav, ["positive electrode"])

        variables.update(
            {
                "X-averaged time derivative of moving phase boundary location [m.s-1]": s_dot,
                "X-averaged lithium stoichiometry at core-shell interface": c_c_b_xav,
                "X-averaged lithium concentration at core-shell interface [mol.m-3]": c_c_b_xav
                * self.param.p.prim.c_max,
                "Lithium stoichiometry at core-shell interface": c_c_b,
                "Lithium concentration at core-shell interface [mol.m-3]": c_c_b
                * self.param.p.prim.c_max,
                "X-averaged oxygen stoichiometry at core-shell interface": c_o_b_xav,
                "X-averaged oxygen concentration at core-shell interface [mol.m-3]": c_o_b_xav
                * self.c_o_core_dim,
                "Oxygen stoichiometry at core-shell interface": c_o_b,
                "Oxygen concentration at core-shell interface [mol.m-3]": c_o_b
                * self.c_o_core_dim,
            }
        )

        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_c_xav = variables["X-averaged positive core stoichiometry"]
        c_o_xav = variables["X-averaged positive shell oxygen stoichiometry"]
        s_xav = variables["X-averaged moving phase boundary location [m]"]

        T_xav = variables["X-averaged positive electrode temperature [K]"]
        T_xav_c = pybamm.PrimaryBroadcast(T_xav, ["positive core"])
        T_xav_o = pybamm.PrimaryBroadcast(T_xav, ["positive shell oxygen"])

        D_c = self.D_c_dimensional(c_c_xav, T_xav_c)
        D_o = self.D_o_dimensional(c_o_xav, T_xav_o)
        R_p = self.param.p.prim.R_typ

        eta = pybamm.positive_electrode_degradation.eta_xav
        psi = pybamm.positive_electrode_degradation.psi_xav

        s_dot = variables[
            "X-averaged time derivative of moving phase boundary location [m.s-1]"
        ]

        self.rhs[c_c_xav] = pybamm.inner(  # Ghosh et al 2021 [C.1]
            eta * s_dot / s_xav, pybamm.grad(c_c_xav)
        ) + 1 / s_xav**2 * pybamm.div(D_c * pybamm.grad(c_c_xav))

        self.rhs[c_o_xav] = pybamm.inner(  # Ghosh et al 2021 [C.3]
            (1 - psi) * s_dot / (R_p - s_xav), pybamm.grad(c_o_xav)
        ) + 1 / ((R_p - s_xav) ** 2 * (psi * (R_p - s_xav) + s_xav) ** 2) * pybamm.div(
            (psi * (R_p - s_xav) + s_xav) ** 2 * D_o * pybamm.grad(c_o_xav)
        )

        self.rhs[s_xav] = s_dot

    def set_boundary_conditions(self, variables):
        c_c_xav = variables["X-averaged positive core stoichiometry"]
        c_o_xav = variables["X-averaged positive shell oxygen stoichiometry"]

        c_c_N_av = variables["X-averaged positive core surface cell stoichiometry"]
        c_o_1_av = variables[
            "X-averaged positive shell center cell oxygen stoichiometry"
        ]
        dx_cp_av = variables["X-averaged positive core surface cell length"]
        dx_co_av = variables["X-averaged positive shell center cell length of oxygen"]

        c_c_b = variables["X-averaged lithium stoichiometry at core-shell interface"]
        c_o_b = variables["X-averaged oxygen stoichiometry at core-shell interface"]

        rbc_cc = (c_c_b - c_c_N_av) / dx_cp_av
        lbc_co = (c_o_1_av - c_o_b) / dx_co_av

        self.boundary_conditions[c_c_xav] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (rbc_cc, "Neumann"),
        }
        self.boundary_conditions[c_o_xav] = {
            "left": (lbc_co, "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }

    def set_initial_conditions(self, variables):
        """
        For single particle models, initial conditions can't depend on x so we
        arbitrarily set the initial values of the single particles to be given
        by the values at x=1 in the positive electrode.
        Typically, supplied initial conditions are uniform x.
        """
        c_c_xav = variables["X-averaged positive core stoichiometry"]
        c_o_xav = variables["X-averaged positive shell oxygen stoichiometry"]
        s_xav = variables["X-averaged moving phase boundary location [m]"]

        eta = pybamm.positive_electrode_degradation.eta_xav * s_xav
        psi = pybamm.positive_electrode_degradation.psi_xav

        self.initial_conditions[c_c_xav] = self.c_c_init(eta)
        self.initial_conditions[c_o_xav] = self.c_o_init(psi)
        self.initial_conditions[s_xav] = self.s_init(self.param.L_x)
