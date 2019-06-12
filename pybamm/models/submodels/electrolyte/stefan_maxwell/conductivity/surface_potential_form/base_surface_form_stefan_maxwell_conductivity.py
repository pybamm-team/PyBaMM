#
# Base class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm


class BaseSurfaceFormStefanMaxwellConductivity(pybamm.BaseSurfaceFormStefanMaxwell):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSurfaceFormStefanMaxwellConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_fundamental_variables(self):
        """
        Returns the variables in the submodel which can be stated independent of
        variables stated in other submodels
        """

        if self._domain == "Negative":
            delta_phi = pybamm.standard_variables.delta_phi_n
        elif self._domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p
        else:
            raise pybamm.DomainError

        delta_phi_av = pybamm.average(delta_phi)

        variables = pybamm.get_standard_surface_potential_difference_variables(
            delta_phi, delta_phi_av
        )

        return variables

    def get_coupled_variables(self, variables):

        if self._domain == "Negative":
            variables.update(self._get_neg_pos_coupled_variables(self, variables))
        elif self._domain == "Separator":
            variables.update(self._get_sep_coupled_variables(self, variables))
        elif self._domain == "Positive":
            variables.update(self._get_neg_pos_coupled_variables(self, variables))
            variables.update(self._get_whole_cell_variables(self, variables))
        else:
            raise pybamm.DomainError

        return variables

    def _get_neg_pos_coupled_variables(self, variables):

        param = self.param

        eps = variables[self._domain + " porosity"]
        c_e = variables[self._domain + " electrolyte concentration"]
        delta_phi = variables[self._domain + " electrode surface potential difference"]
        phi_s = variables[self._domain + " electrode potential"]

        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e

        i_e = conductivity * (
            (param.chi(c_e) / c_e) * pybamm.grad(c_e) + pybamm.grad(delta_phi)
        )

        phi_e = phi_s - delta_phi  # TODO: make this not require phi_s

        variables.update(self._get_domain_potential_variables(phi_e))
        variables.update(self._get_domain_current_variables(i_e))

        return variables

    def _get_sep_coupled_variables(self, variables):

        param = self.param
        x_s = pybamm.standard_spatial_vars.x_s

        i_boundary_cc = variables["Current collector current density"]
        c_e_s = variables["Separator electrolyte concentration"]
        phi_e_n = variables["Negative electrode surface potential difference"]
        eps_s = variables["Separator porosity"]

        chi_e_s = param.chi(c_e_s)
        kappa_s_eff = param.kappa_e(c_e_s) * (eps_s ** param.b)
        i_e_s = i_boundary_cc

        phi_e_s = pybamm.boundary_value(phi_e_n, "right") + pybamm.IndefiniteIntegral(
            chi_e_s / c_e_s * pybamm.grad(c_e_s) - param.C_e * i_e_s / kappa_s_eff, x_s
        )

        variables.update(self._get_domain_potential_variables(phi_e_s))
        variables.update(self._get_domain_current_variables(i_e_s))

        return variables

    def set_boundary_conditions(self, variables):
        """
        Set boundary conditions for the full model.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """
        param = self.set_of_parameters

        i_boundary_cc = variables["Current collector current density"]
        eps = variables[self._domain + " porosity"]
        c_e = variables[self._domain + " electrolyte concentration"]
        delta_phi = variables[self._domain + " electrode surface potential difference"]

        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e

        if self._domain == "Negative":

            lbc = (pybamm.Scalar(0), "Neumann")

            c_e_flux = pybamm.BoundaryFlux(c_e, "right")
            flux = (
                i_boundary_cc / pybamm.BoundaryValue(conductivity, "right")
            ) - pybamm.BoundaryValue(param.chi(c_e) / c_e, "right") * c_e_flux
            rbc = (flux, "Neumann")

        elif self._domain == "Positive":

            c_e_flux = pybamm.BoundaryFlux(c_e, "left")
            flux = (
                i_boundary_cc / pybamm.BoundaryValue(conductivity, "left")
            ) - pybamm.BoundaryValue(param.chi(c_e) / c_e, "left") * c_e_flux
            lbc = (flux, "Neumann")

            rbc = (pybamm.Scalar(0), "Neumann")

        else:
            raise pybamm.DomainError

        self.boundary_conditions = {delta_phi: {"left": lbc, "right": rbc}}

    def set_initial_conditions(self, variables):
        """Initial condition"""
        delta_phi_e = variables[
            self._domain + " electrode surface potential difference"
        ]
        if self._domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init)
        elif self._domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init)

        else:
            raise pybamm.DomainError

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}

