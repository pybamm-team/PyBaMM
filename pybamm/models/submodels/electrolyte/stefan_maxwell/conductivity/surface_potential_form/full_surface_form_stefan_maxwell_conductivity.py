#
# Class for electrolyte conductivity employing stefan-maxwell
#
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
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.BaseModel`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def get_fundamental_variables(self):
        if self.domain == "Negative":
            delta_phi = pybamm.standard_variables.delta_phi_n
        elif self.domain == "Separator":
            return {}
        elif self.domain == "Positive":
            delta_phi = pybamm.standard_variables.delta_phi_p
        else:
            raise pybamm.DomainError

        variables = self._get_standard_surface_potential_difference_variables(delta_phi)

        return variables

    def set_initial_conditions(self, variables):
        if self.domain == "Separator":
            return

        delta_phi_e = variables[self.domain + " electrode surface potential difference"]
        if self.domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init)
        elif self.domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init)
        else:
            raise pybamm.DomainError

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}

    def get_coupled_variables(self, variables):

        if self.domain == "Negative":
            variables.update(self._get_neg_pos_coupled_variables(variables))
        elif self.domain == "Separator":
            variables.update(self._get_sep_coupled_variables(variables))
        elif self.domain == "Positive":
            variables.update(self._get_neg_pos_coupled_variables(variables))
            variables.update(self._get_whole_cell_variables(variables))

        return variables

    def set_boundary_conditions(self, variables):
        if self.domain == "Separator":
            return None

        param = self.param

        i_boundary_cc = variables["Current collector current density"]
        eps = variables[self.domain + " electrode porosity"]
        c_e = variables[self.domain + " electrolyte concentration"]
        delta_phi = variables[self.domain + " electrode surface potential difference"]

        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e

        if self.domain == "Negative":

            lbc = (pybamm.Scalar(0), "Neumann")

            c_e_flux = pybamm.BoundaryFlux(c_e, "right")

            flux = (
                i_boundary_cc / pybamm.BoundaryValue(conductivity, "right")
            ) - pybamm.BoundaryValue(param.chi(c_e) / c_e, "right") * c_e_flux

            rbc = (flux, "Neumann")

        elif self.domain == "Positive":

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
        """
        A private function to get the coupled variables when the domain is 'Negative'
        or 'Positive'.
        """

        param = self.param

        eps = variables[self.domain + " electrode porosity"]
        c_e = variables[self.domain + " electrolyte concentration"]
        delta_phi = variables[self.domain + " electrode surface potential difference"]

        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e

        # This is a bit of a hack until we figure out how we want to take gradients of
        # non-state variables (i.e. put the bcs on without bcs)
        if c_e.has_symbol_of_class(pybamm.Broadcast):
            grad_c_e = pybamm.Broadcast(0, [self.domain.lower() + " electrode"])
        else:
            grad_c_e = pybamm.grad(c_e)
            grad_c_e = pybamm.Broadcast(0, [self.domain.lower() + " electrode"])

        i_e = conductivity * (
            (param.chi(c_e) / c_e) * grad_c_e + pybamm.grad(delta_phi)
        )

        # TODO: Expression can be written in a form which does not require phi_s and
        # so avoid this hack.
        phi_s = self.nasty_hack_to_get_phi_s(variables)
        phi_e = phi_s - delta_phi

        variables.update(self._get_domain_potential_variables(phi_e))
        variables.update(self._get_domain_current_variables(i_e))

        return variables

    def _get_sep_coupled_variables(self, variables):
        """
        A private function to get the coupled variables when the domain is 'Separator'.
        """

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
            grad_c_e_s = pybamm.Broadcast(0, ["separator"])

        phi_e_s = pybamm.boundary_value(phi_e_n, "right") + pybamm.IndefiniteIntegral(
            chi_e_s / c_e_s * grad_c_e_s - param.C_e * i_e_s_av / kappa_s_eff, x_s
        )

        i_e_s = pybamm.Broadcast(i_e_s_av, ["separator"])
        phi_e_s = pybamm.Broadcast(
            pybamm.boundary_value(phi_e_n, "right"), ["separator"]
        )  # TODO: add Indefinite integral!

        variables.update(self._get_domain_potential_variables(phi_e_s))
        variables.update(self._get_domain_current_variables(i_e_s))

        return variables

    def nasty_hack_to_get_phi_s(self, variables):
        "This restates what is already in the electrode submodel which we should not do"

        param = self.param

        x_n = pybamm.standard_spatial_vars.x_n
        x_p = pybamm.standard_spatial_vars.x_p
        eps = variables[self.domain + " electrode porosity"]
        c_e = variables[self.domain + " electrolyte concentration"]
        delta_phi = variables[self.domain + " electrode surface potential difference"]
        conductivity = param.kappa_e(c_e) * (eps ** param.b) / param.C_e / param.gamma_e
        i_boundary_cc = variables["Current collector current density"]

        # This is a bit of a hack until we figure out how we want to take gradients of
        # non-state variables (i.e. put the bcs on without bcs)
        # and set internal boundary conditions
        if c_e.has_symbol_of_class(pybamm.Broadcast):
            grad_c_e = pybamm.Broadcast(0, [self.domain.lower() + " electrode"])
        else:
            grad_c_e = pybamm.grad(c_e)
            grad_c_e = pybamm.Broadcast(0, [self.domain.lower() + " electrode"])

        i_e = conductivity * (
            (param.chi(c_e) / c_e) * grad_c_e + pybamm.grad(delta_phi)
        )

        i_s = i_boundary_cc - i_e

        if self.domain == "Negative":
            conductivity = param.sigma_n * (1 - eps) ** param.b
            phi_s = -pybamm.IndefiniteIntegral(i_s / conductivity, x_n)

        elif self.domain == "Positive":

            phi_e_s = variables["Separator electrolyte potential"]
            delta_phi_p = variables["Positive electrode surface potential difference"]

            conductivity = param.sigma_p * (1 - eps) ** param.b
            phi_s = (
                -pybamm.IndefiniteIntegral(i_s / conductivity, x_p)
                + pybamm.boundary_value(phi_e_s, "right")
                + pybamm.boundary_value(delta_phi_p, "left")
            )

        return phi_s


class FullAlgebraic(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


     **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseFull`
    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):
        if self.domain == "Separator":
            return

        delta_phi = variables[self.domain + " electrode surface potential difference"]
        i_e = variables[self.domain + " electrolyte current density"]
        j = variables[self.domain + " electrode interfacial current density"]

        self.rhs[delta_phi] = pybamm.div(i_e) - j


class FullDifferential(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations and where capacitance is present.
    (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte.stefan_maxwell.conductivity.surface_potential_form.BaseFull`

    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_rhs(self, variables):
        if self.domain == "Separator":
            return

        if self.domain == "Negative":
            C_dl = self.param.C_dl_n
        elif self.domain == "Positive":
            C_dl = self.param.C_dl_p

        delta_phi = variables[self.domain + " electrode surface potential difference"]
        i_e = variables[self.domain + " electrolyte current density"]
        j = variables[self.domain + " electrode interfacial current density"]

        self.rhs[delta_phi] = 1 / C_dl * (pybamm.div(i_e) - j)
