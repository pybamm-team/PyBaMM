#
# Base class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm

from .base_surface_form_stefan_maxwell_conductivity import (
    BaseSurfaceFormStefanMaxwellConductivity,
)


class BaseFullSurfaceFormStefanMaxwellConductivity(
    BaseSurfaceFormStefanMaxwellConductivity
):
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

