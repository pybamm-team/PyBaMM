#
# Full model for Ohm's law in the electrode
#
import pybamm


class FullOhm(pybamm.BaseOhm):
    """Full model for ohm's law with conservation of current for the current in the 
    electrodes.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'Negative electrode' or 'Positive electrode'

    *Extends:* :class:`pybamm.BaseOhm`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel for which a PDE must be solved to obtain.
        """

        if self._domain == "Negative":
            phi_s = pybamm.standard_variables.phi_s_n
        elif self._domain == "Positive":
            phi_s = pybamm.standard_variables.phi_s_p
        else:
            pybamm.DomainError("Domain must be either: 'Negative' or 'Positive'")

        fundamental_variables = {self._domain + " electrode potential": phi_s}

        return fundamental_variables

    def get_derived_variables(self, variables):
        """
        Returns variables which are derived from the fundamental variables in the model.
        """

        phi_s = variables[self._domain + " electrode potential"]
        eps = variables[self._domain + " electrode porosity"]

        if self._domain == "Negative":
            sigma = self.param.sigma_n
        elif self._domain == "Positive":
            sigma = self.param.sigma_p

        sigma_eff = sigma * (1 - eps)
        i_s = -sigma_eff * pybamm.grad(phi_s)

        derived_variables = {
            self._domain + " electrode current density": i_s,
            self._domain + " electrode effective conductivity": sigma_eff,
        }

        derived_variables = self.standard_derived_variables(derived_variables)

        return derived_variables

    def set_algebraic(self, variables):
        """
        PDE for current in the electrodes, using Ohm's law

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " electrode potential"]
        i_s = variables[self._domain + " electrode current density"]
        j = variables[self._domain + " interfacial current density"]

        self.algebraic[phi_s] = pybamm.div(i_s) + j

    def set_boundary_conditions(self, variables):
        """
        Boundary conditions for current in the electrodes.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " electrode potential"]
        sigma_eff = variables[self._domain + " electrode effective conductivity"]
        i_boundary_cc = variables["Current collector current density"]

        if self._domain == "Negative":
            lbc = (pybamm.Scalar(0), "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")

        elif self._domain == "Positive":
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (
                i_boundary_cc
                / pybamm.boundary_value(-sigma_eff ** self.param.b, "right"),
                "Neumann",
            )

        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}

    def set_initial_conditions(self, variables):
        """
        Initial conditions for current and potentials in the electrodes.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        phi_s = variables[self._domain + " electrode potential"]

        if self._domain == "Negative":
            phi_s_init = pybamm.Scalar(0)
        elif self._domain == "Positive":
            phi_s_init = self.param.U_p(self.param.c_p_init) - self.param.U_n(
                self.param.c_n_init
            )

        self.initial_conditions[phi_s] = phi_s_init

    @property
    def default_solver(self):
        """
        Create and return the default solver for this model
        """
        return pybamm.ScikitsDaeSolver()
