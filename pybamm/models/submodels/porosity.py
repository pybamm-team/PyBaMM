#
# Equation classes for the electrolyte porosity
#
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

    def set_differential_system(self, variables):
        """
        ODE system for the change in porosity due to reactions

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        epsilon = variables["Porosity"]
        j = variables["Interfacial current density"]
        param = self.set_of_parameters

        deps_dt = -param.beta_surf * j
        self.rhs = {epsilon: deps_dt}
        self.initial_conditions = {epsilon: param.eps_init}

        self.variables = {"Porosity": epsilon, "Porosity change": deps_dt}

        # Terminate if porosity reaches zero or one
        self.events = {
            "Minimum porosity cut-off": pybamm.min(epsilon),
            "Maximum porosity cut-off": pybamm.max(epsilon) - 1,
        }

    def set_leading_order_system(self, variables):
        """
        ODE system for the leading-order change in porosity due to reactions

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        param = self.set_of_parameters
        epsilon = variables["Porosity"]
        eps_n, eps_s, eps_p = [e.orphans[0] for e in epsilon.orphans]
        j_n = variables[
            "Average negative electrode interfacial current density per volume"
        ]
        j_s = pybamm.Scalar(0)
        j_p = variables[
            "Average positive electrode interfacial current density per volume"
        ]

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
            # Terminate if porosity reaches zero or one
            cutoff = "porosity cut-off (" + domain + ")"
            self.events.update(
                {
                    "Minimum " + cutoff: pybamm.min(epsilon),
                    "Maximum " + cutoff: pybamm.max(epsilon) - 1,
                }
            )

        deps_dt = pybamm.Concatenation(
            self.variables["Negative electrode porosity change"],
            self.variables["Separator porosity change"],
            self.variables["Positive electrode porosity change"],
        )
        self.variables.update({"Porosity change": deps_dt})
