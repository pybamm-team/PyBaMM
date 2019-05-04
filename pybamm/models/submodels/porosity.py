#
# Equation classes for the electrolyte porosity
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import pybamm


class Standard(pybamm.SubModel):
    """Change in porosity due to reactions

    Parameters
    ----------
    set_of_parameters : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.SubModel`
    """

    def __init__(self, set_of_parameters):
        super().__init__(set_of_parameters)

    def set_differential_system(self, epsilon, j_n, j_p):
        """
        ODE system for the change in porosity due to reactions

        Parameters
        ----------
        epsilon : :class:`pybamm.Symbol`
            The porosity variable
        j_n : :class:`pybamm.Symbol`
            Interfacial current density in the negative electrode
        j_p : :class:`pybamm.Symbol`
            Interfacial current density in the positive electrode
        """
        param = self.set_of_parameters

        j = pybamm.Concatenation(j_n, pybamm.Broadcast(0, ["separator"]), j_p)
        deps_dt = -param.beta_surf * j
        self.rhs = {epsilon: deps_dt}
        self.initial_conditions = {epsilon: param.eps_init}

        eps_n, eps_s, eps_p = epsilon.orphans
        self.variables = {
            "Porosity": epsilon,
            "Negative electrode porosity": eps_n,
            "Separator porosity": eps_s,
            "Positive electrode porosity": eps_p,
            "Porosity change": deps_dt,
        }

    def set_leading_order_system(self, epsilon, j_n, j_p):
        """
        ODE system for the leading-order change in porosity due to reactions
        Parameters
        ----------
        epsilon : :class:`pybamm.Concatenation`
            The porosity variable
        j_n : :class:`pybamm.Symbol`
            Interfacial current density in the negative electrode
        j_p : :class:`pybamm.Symbol`
            Interfacial current density in the positive electrode
        """
        param = self.set_of_parameters

        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]
        j_s = pybamm.Scalar(0)

        self.variables = {"Porosity": epsilon}
        self.leading_order_variables = {}
        for (eps, j, beta_surf, eps_init, domain) in [
            (eps_n, j_n, param.beta_surf_n, param.eps_n_init, "negative electrode"),
            (eps_s, j_s, 0, param.eps_s_init, "separator"),
            (eps_p, j_p, param.beta_surf_p, param.eps_p_init, "positive electrode"),
        ]:
            Domain = domain.capitalize()

            # Model
            deps_dt = -beta_surf * j
            self.rhs.update({eps: deps_dt})
            self.initial_conditions.update({eps: eps_init})
            self.variables.update(
                {
                    Domain + " porosity": pybamm.Broadcast(eps, domain),
                    Domain + " porosity change": pybamm.Broadcast(deps_dt, domain),
                }
            )
