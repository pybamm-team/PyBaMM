#
# Base class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm
from ..base_stefan_maxwell_conductivity import (
    BaseModel as BaseStefanMaxwellConductivity,
)


class BaseModel(BaseStefanMaxwellConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation. (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseStefanMaxwellConductivity`
    """

    def __init__(self, param, domain):
        super().__init__(param)
        self._domain = domain

    def get_coupled_variables(self, variables):

        if self._domain == "Negative":
            variables.update(self._get_neg_pos_coupled_variables(variables))
        elif self._domain == "Separator":
            variables.update(self._get_sep_coupled_variables(variables))
        elif self._domain == "Positive":
            variables.update(self._get_neg_pos_coupled_variables(variables))
            variables.update(self._get_whole_cell_variables(variables))
        else:
            raise pybamm.DomainError

        return variables

    def set_boundary_conditions(self, variables):
        """
        Set boundary conditions for the full model.

        Parameters
        ----------
        variables : dict
            Dictionary of symbols to use in the model
        """

        if self._domain == "Separator":
            return None

        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        eps = variables[self._domain + " electrode porosity"]
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

    def _get_neg_pos_coupled_variables(self, variables):

        param = self.param

        eps = variables[self._domain + " electrode porosity"]
        c_e = variables[self._domain + " electrolyte concentration"]
        delta_phi = variables[self._domain + " electrode surface potential difference"]
        phi_s = variables[self._domain + " electrode potential"]

        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e

        # This is a bit of a hack until we figure out how we want to take gradients of
        # non-state variables (i.e. put the bcs on without bcs)
        if c_e.has_symbol_of_class(pybamm.Broadcast):
            grad_c_e = pybamm.Broadcast(0, [self._domain.lower() + " electrode"])
        else:
            grad_c_e = pybamm.grad(c_e)

        i_e = conductivity * (
            (param.chi(c_e) / c_e) * grad_c_e + pybamm.grad(delta_phi)
        )

        # TODO: Expression can be written in a form which does not require phi_s and
        # so avoid this hack.
        phi_s = self.nasty_hack_to_get_phi_s(variables)
        phi_e = phi_s - delta_phi

        variables.update(self._get_domain_potential_variables(phi_e, self._domain))
        variables.update(self._get_domain_current_variables(i_e, self._domain))

        return variables

    def _get_sep_coupled_variables(self, variables):

        param = self.param
        x_s = pybamm.standard_spatial_vars.x_s

        i_boundary_cc = variables["Current collector current density"]
        c_e_s = variables["Separator electrolyte concentration"]
        phi_e_n = variables["Negative electrolyte potential"]
        eps_s = variables["Separator porosity"]

        chi_e_s = param.chi(c_e_s)
        kappa_s_eff = param.kappa_e(c_e_s) * (eps_s ** param.b)
        i_e_s_av = i_boundary_cc

        # This is a bit of a hack until we figure out how we want to take gradients of
        # non-state variables (i.e. put the bcs on without bcs)
        if c_e_s.has_symbol_of_class(pybamm.Broadcast):
            grad_c_e_s = pybamm.Broadcast(0, ["separator"])
        else:
            grad_c_e_s = pybamm.grad(c_e_s)

        phi_e_s = pybamm.boundary_value(phi_e_n, "right") + pybamm.IndefiniteIntegral(
            chi_e_s / c_e_s * grad_c_e_s - param.C_e * i_e_s_av / kappa_s_eff, x_s
        )

        i_e_s = pybamm.Broadcast(i_e_s_av, ["separator"])
        phi_e_s = pybamm.Broadcast(
            pybamm.boundary_value(phi_e_n, "right"), ["separator"]
        )  # TODO: add Indefinite integral!

        variables.update(self._get_domain_potential_variables(phi_e_s, self._domain))
        variables.update(self._get_domain_current_variables(i_e_s, self._domain))

        return variables

    def nasty_hack_to_get_phi_s(self, variables):
        "This restates what is already in the electrode submodel which we should not do"

        param = self.param

        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        eps = variables[self._domain + " electrode porosity"]
        c_e = variables[self._domain + " electrolyte concentration"]
        delta_phi = variables[self._domain + " electrode surface potential difference"]
        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
        i_boundary_cc = variables["Current collector current density"]

        # This is a bit of a hack until we figure out how we want to take gradients of
        # non-state variables (i.e. put the bcs on without bcs)
        if c_e.has_symbol_of_class(pybamm.Broadcast):
            grad_c_e = pybamm.Broadcast(0, [self._domain.lower() + " electrode"])
        else:
            grad_c_e = pybamm.grad(c_e)

        i_e = conductivity * (
            (param.chi(c_e) / c_e) * grad_c_e + pybamm.grad(delta_phi)
        )

        i_s = i_boundary_cc - i_e

        if self._domain == "Negative":
            conductivity = param.sigma_n * (1 - eps) ** param.b
            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_n)

        elif self._domain == "Positive":

            phi_e_s = variables["Separator electrolyte potential"]
            delta_phi_p = variables["Positive electrode surface potential difference"]

            conductivity = param.sigma_p * (1 - eps) ** param.b
            phi_s = (
                -pybamm.IndefiniteIntegral(i_s / conductivity, x_p)
                + pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        return phi_s

