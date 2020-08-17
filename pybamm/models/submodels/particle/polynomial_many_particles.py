#
# Class for many particles with polynomial concentration profile
#
import pybamm

from .base_particle import BaseParticle


class PolynomialManyParticles(BaseParticle):
    """
    Base class for molar conservation in many particles with an assumed polynomial
    concentration profile in r.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    order : int, optional
        The order of the polynomial, can be 2.


    **Extends:** :class:`pybamm.particle.BaseParticle`
    """

    def __init__(self, param, domain, order=2):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            c_s_rav = pybamm.standard_variables.c_s_n_rav
            c_s_surf = pybamm.standard_variables.c_s_n_surf
            r = pybamm.standard_spatial_vars.r_n

        elif self.domain == "Positive":
            c_s_rav = pybamm.standard_variables.c_s_p_rav
            c_s_surf = pybamm.standard_variables.c_s_p_surf
            r = pybamm.standard_spatial_vars.r_p

        # The concentration is given by c = A + B*r**2
        A = pybamm.PrimaryBroadcast(
            (1 / 2) * (5 * c_s_rav - 3 * c_s_surf), [self.domain.lower() + " particle"]
        )
        B = pybamm.PrimaryBroadcast(
            (5 / 2) * (c_s_surf - c_s_rav), [self.domain.lower() + " particle"]
        )
        c_s = A + B * r ** 2

        variables = self._get_standard_concentration_variables(
            c_s, c_s_rav=c_s_rav, c_s_surf=c_s_surf
        )

        return variables

    def get_coupled_variables(self, variables):
        c_s = variables[self.domain + " particle concentration"]
        T_k = pybamm.PrimaryBroadcast(
            variables[self.domain + " electrode temperature"],
            [self.domain.lower() + " particle"],
        )

        if self.domain == "Negative":
            N_s = -self.param.D_n(c_s, T_k) * pybamm.grad(c_s)
            x = pybamm.standard_spatial_vars.x_n
            R = self.param.R_n_of_x(x)
            variables.update({"Negative particle distribution in x": R})

        elif self.domain == "Positive":
            N_s = -self.param.D_p(c_s, T_k) * pybamm.grad(c_s)

            x = pybamm.standard_spatial_vars.x_p
            R = self.param.R_p_of_x(x)
            variables.update({"Positive particle distribution in x": R})

        variables.update(self._get_standard_flux_variables(N_s, N_s))

        return variables

    def set_rhs(self, variables):
        c_s_rav = variables[
            "R-averaged " + self.domain.lower() + " particle concentration"
        ]
        j = variables[self.domain + " electrode interfacial current density"]
        R = variables[self.domain + " particle distribution in x"]

        if self.domain == "Negative":
            self.rhs = {c_s_rav: -3 * j / self.param.a_n / R}

        elif self.domain == "Positive":
            self.rhs = {c_s_rav: -3 * j / self.param.a_p / self.param.gamma_p / R}

    def set_algebraic(self, variables):
        c_s_surf = variables[self.domain + " particle surface concentration"]
        c_s_rav = variables[
            "R-averaged " + self.domain.lower() + " particle concentration"
        ]
        j = variables[self.domain + " electrode interfacial current density"]
        T = variables[self.domain + " electrode temperature"]
        R = variables[self.domain + " particle distribution in x"]

        if self.domain == "Negative":
            self.algebraic = {
                c_s_surf: c_s_surf
                - c_s_rav
                + self.param.C_n
                * (j * R / 5 / self.param.a_n / self.param.D_n(c_s_surf, T))
            }

        elif self.domain == "Positive":
            self.algebraic = {
                c_s_surf: c_s_surf
                - c_s_rav
                + self.param.C_p
                * (
                    j
                    * R
                    / 5
                    / self.param.a_p
                    / self.param.gamma_p
                    / self.param.D_p(c_s_surf, T)
                )
            }

    def set_initial_conditions(self, variables):
        c_s_rav = variables[
            "R-averaged " + self.domain.lower() + " particle concentration"
        ]
        c_s_surf = variables[self.domain + " particle surface concentration"]

        if self.domain == "Negative":
            x_n = pybamm.standard_spatial_vars.x_n
            c_init = self.param.c_n_init(x_n)

        elif self.domain == "Positive":
            x_p = pybamm.standard_spatial_vars.x_p
            c_init = self.param.c_p_init(x_p)

        self.initial_conditions = {c_s_rav: c_init, c_s_surf: c_init}
