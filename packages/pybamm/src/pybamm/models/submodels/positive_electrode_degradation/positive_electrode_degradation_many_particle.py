#
# Class for many particles (DFN/P2D) with positive electrode degradation
#

import pybamm

from .base_positive_electrode_degradation import BasePositiveElectrodeDegradation


class PositiveElectrodeDegradationManyParticle(BasePositiveElectrodeDegradation):
    """
    Class for many particles (DFN/P2D) with positive electrode degradation

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
        pybamm.citations.register("Zhuo2023")

    def get_fundamental_variables(self):
        # Concentration in particle core
        c_c = pybamm.Variable(
            "Positive core stoichiometry",
            domains={
                "primary": "positive core",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
            bounds=(0, 1),
        )

        # Oxygen concentration in degraded passivation layer/shell
        c_o = pybamm.Variable(
            "Positive shell oxygen stoichiometry",
            domains={
                "primary": "positive shell oxygen",
                "secondary": "positive electrode",
                "tertiary": "current collector",
            },
        )

        # location of core-shell boundary, one value per through-cell position.
        s = pybamm.Variable(
            "Moving phase boundary location [m]",
            domains={
                "primary": "positive electrode",
                "secondary": "current collector",
            },
            bounds=(0, self.param.p.prim.R_typ),  # s cannot exceed the particle radius
        )

        variables = self._get_standard_concentration_variables(c_c=c_c, c_o=c_o, s=s)

        return variables

    def get_coupled_variables(self, variables):

        T = variables["Positive electrode temperature [K]"]

        c_o_cent = variables["Positive shell center oxygen stoichiometry"]
        c_c_surf = variables["Positive core surface stoichiometry"]

        R = variables["Positive particle radius [m]"]

        # Get s_dot
        K_1 = self.k_1_dimensional(T)  # [m.s-1]
        K_2 = self.k_2_dimensional(T) * self.c_o_core_dim  # [m.s-2]
        s_dot = -(  # Zhuo et al 2023 (13), Ghosh et al 2021 [C.12]
            K_1 - K_2 * c_o_cent
        ) * pybamm.EqualHeaviside(c_c_surf, self.c_p_thrd)

        # Get c_c and c_o at boundary
        c_c = variables["Positive core stoichiometry"]
        c_o = variables["Positive shell oxygen stoichiometry"]
        s = variables["Moving phase boundary location [m]"]

        T_c = pybamm.PrimaryBroadcast(T, ["positive core"])
        T_o = pybamm.PrimaryBroadcast(T, ["positive shell oxygen"])

        D_c = pybamm.surf(self.D_c_dimensional(c_c, T_c))
        D_o = pybamm.boundary_value(self.D_o_dimensional(c_o, T_o), "left")

        c_c_N = variables["Positive core surface cell stoichiometry"]
        c_o_1 = variables["Positive shell center cell oxygen stoichiometry"]
        dx_cp = variables["Positive core surface cell length"]
        dx_co = variables["Positive shell center cell length of oxygen"]

        j = variables["Positive electrode interfacial current density [A.m-2]"]

        # The interface boundary value is calculated from the applied boundary
        # condition, not extrapolated afterwards.

        F = self.param.F
        c_p_max = self.param.p.prim.c_max

        # Boundary core lithium concentration from applied bc
        d = D_c / (s * dx_cp)
        J = (R / s) ** 2 * (j / F) / c_p_max
        c_c_b = ((s_dot * self.c_s_trap) + (c_c_N * d) - J) / (
            s_dot + d
        )  # Solved from Zhuo (10)  & (A.2)
        c_c_b_xav = pybamm.x_average(c_c_b)

        # Boundary oxygen concentration from applied bc
        d_o = D_o / ((R - s) * dx_co)
        c_o_b = ((c_o_1 * d_o) - (s_dot * self.c_o_core)) / (
            d_o - s_dot
        )  # Solved from Zhuo (12)
        c_o_b_xav = pybamm.x_average(c_o_b)

        variables.update(
            {
                "Time derivative of moving phase boundary location [m.s-1]": s_dot,
                "X-averaged time derivative of moving phase boundary location [m.s-1]": (
                    pybamm.x_average(s_dot)
                ),
                "Lithium stoichiometry at core-shell interface": c_c_b,
                "Lithium concentration at core-shell interface [mol.m-3]": c_c_b
                * self.param.p.prim.c_max,
                "X-averaged lithium stoichiometry at core-shell interface": c_c_b_xav,
                "X-averaged lithium concentration at core-shell interface [mol.m-3]": c_c_b_xav
                * self.param.p.prim.c_max,
                "Oxygen stoichiometry at core-shell interface": c_o_b,
                "Oxygen concentration at core-shell interface [mol.m-3]": c_o_b
                * self.c_o_core_dim,
                "X-averaged oxygen stoichiometry at core-shell interface": c_o_b_xav,
                "X-averaged oxygen concentration at core-shell interface [mol.m-3]": c_o_b_xav
                * self.c_o_core_dim,
            }
        )

        variables.update(self._get_total_concentration_variables(variables))

        return variables

    def set_rhs(self, variables):
        c_c = variables["Positive core stoichiometry"]
        c_o = variables["Positive shell oxygen stoichiometry"]
        s = variables["Moving phase boundary location [m]"]
        R = variables["Positive particle radius [m]"]

        T = variables["Positive electrode temperature [K]"]
        T_c = pybamm.PrimaryBroadcast(T, ["positive core"])
        T_o = pybamm.PrimaryBroadcast(T, ["positive shell oxygen"])

        D_c = self.D_c_dimensional(c_c, T_c)
        D_o = self.D_o_dimensional(c_o, T_o)

        eta = pybamm.positive_electrode_degradation.eta
        psi = pybamm.positive_electrode_degradation.psi

        s_dot = variables["Time derivative of moving phase boundary location [m.s-1]"]

        self.rhs[c_c] = pybamm.inner(  # Ghosh et al 2021 [C.1]
            eta * s_dot / s, pybamm.grad(c_c)
        ) + 1 / s**2 * pybamm.div(D_c * pybamm.grad(c_c))

        self.rhs[c_o] = pybamm.inner(  # Ghosh et al 2021 [C.3]
            (1 - psi) * s_dot / (R - s), pybamm.grad(c_o)
        ) + 1 / ((R - s) ** 2 * (psi * (R - s) + s) ** 2) * pybamm.div(
            (psi * (R - s) + s) ** 2 * D_o * pybamm.grad(c_o)
        )

        self.rhs[s] = s_dot

    def set_boundary_conditions(self, variables):
        c_c = variables["Positive core stoichiometry"]
        c_o = variables["Positive shell oxygen stoichiometry"]

        c_c_N = variables["Positive core surface cell stoichiometry"]
        c_o_1 = variables["Positive shell center cell oxygen stoichiometry"]
        dx_cp = variables["Positive core surface cell length"]
        dx_co = variables["Positive shell center cell length of oxygen"]

        c_c_b = variables["Lithium stoichiometry at core-shell interface"]
        c_o_b = variables["Oxygen stoichiometry at core-shell interface"]

        rbc_cc = (c_c_b - c_c_N) / dx_cp
        lbc_co = (c_o_1 - c_o_b) / dx_co

        self.boundary_conditions[c_c] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (rbc_cc, "Neumann"),
        }
        self.boundary_conditions[c_o] = {
            "left": (lbc_co, "Neumann"),
            "right": (pybamm.Scalar(0), "Dirichlet"),
        }

    def set_initial_conditions(self, variables):
        c_c = variables["Positive core stoichiometry"]
        c_o = variables["Positive shell oxygen stoichiometry"]
        s = variables["Moving phase boundary location [m]"]

        x_p = pybamm.standard_spatial_vars.x_p

        eta = pybamm.positive_electrode_degradation.eta
        psi = pybamm.positive_electrode_degradation.psi

        self.initial_conditions[c_c] = self.c_c_init(eta)
        self.initial_conditions[c_o] = self.c_o_init(psi)
        self.initial_conditions[s] = self.s_init(x_p)
