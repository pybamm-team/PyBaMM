#
# Class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm
from ..base_electrolyte_conductivity import BaseElectrolyteConductivity


class BaseModel(BaseElectrolyteConductivity):
    """Base class for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations employing the surface potential difference
    formulation.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    reactions : dict
        Dictionary of reaction terms

    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
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

        variables = self._get_standard_surface_potential_difference_variables(delta_phi)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param

        if self.domain in ["Negative", "Positive"]:

            conductivity, sigma_eff = self._get_conductivities(variables)
            i_boundary_cc = variables["Current collector current density"]
            c_e = variables[self.domain + " electrolyte concentration"]
            delta_phi = variables[
                self.domain + " electrode surface potential difference"
            ]
            T = variables[self.domain + " electrode temperature"]

            i_e = conductivity * (
                ((1 + param.Theta * T) * param.chi(c_e) / c_e) * pybamm.grad(c_e)
                + pybamm.grad(delta_phi)
                + i_boundary_cc / sigma_eff
            )
            variables.update(self._get_domain_current_variables(i_e))

            phi_s = variables[self.domain + " electrode potential"]
            phi_e = phi_s - delta_phi

            variables.update(self._get_domain_potential_variables(phi_e))

        elif self.domain == "Separator":
            x_s = pybamm.standard_spatial_vars.x_s

            i_boundary_cc = variables["Current collector current density"]
            c_e_s = variables["Separator electrolyte concentration"]
            phi_e_n = variables["Negative electrolyte potential"]
            tor_s = variables["Separator porosity"]
            T = variables["Separator temperature"]

            chi_e_s = param.chi(c_e_s)
            kappa_s_eff = param.kappa_e(c_e_s, T) * tor_s

            phi_e_s = pybamm.boundary_value(
                phi_e_n, "right"
            ) + pybamm.IndefiniteIntegral(
                (1 + param.Theta * T) * chi_e_s / c_e_s * pybamm.grad(c_e_s)
                - param.C_e * i_boundary_cc / kappa_s_eff,
                x_s,
            )

            i_e_s = pybamm.PrimaryBroadcast(i_boundary_cc, "separator")

            variables.update(self._get_domain_potential_variables(phi_e_s))
            variables.update(self._get_domain_current_variables(i_e_s))

            # Update boundary conditions (for indefinite integral)
            self.boundary_conditions[c_e_s] = {
                "left": (pybamm.BoundaryGradient(c_e_s, "left"), "Neumann"),
                "right": (pybamm.BoundaryGradient(c_e_s, "right"), "Neumann"),
            }

        if self.domain == "Positive":
            variables.update(self._get_whole_cell_variables(variables))

        return variables

    def _get_conductivities(self, variables):
        param = self.param
        tor_e = variables[self.domain + " electrolyte tortuosity"]
        tor_s = variables[self.domain + " electrode tortuosity"]
        c_e = variables[self.domain + " electrolyte concentration"]
        T = variables[self.domain + " electrode temperature"]
        if self.domain == "Negative":
            sigma = param.sigma_n
        elif self.domain == "Positive":
            sigma = param.sigma_p

        kappa_eff = param.kappa_e(c_e, T) * tor_e
        sigma_eff = sigma * tor_s
        conductivity = kappa_eff / (param.C_e / param.gamma_e + kappa_eff / sigma_eff)

        return conductivity, sigma_eff

    def set_initial_conditions(self, variables):
        if self.domain == "Separator":
            return

        delta_phi_e = variables[self.domain + " electrode surface potential difference"]
        if self.domain == "Negative":
            delta_phi_e_init = self.param.U_n(self.param.c_n_init(0), self.param.T_init)
        elif self.domain == "Positive":
            delta_phi_e_init = self.param.U_p(self.param.c_p_init(1), self.param.T_init)

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}

    def set_boundary_conditions(self, variables):
        if self.domain == "Separator":
            return None

        param = self.param

        conductivity, sigma_eff = self._get_conductivities(variables)
        i_boundary_cc = variables["Current collector current density"]
        c_e = variables[self.domain + " electrolyte concentration"]
        delta_phi = variables[self.domain + " electrode surface potential difference"]

        if self.domain == "Negative":
            T = variables["Negative electrode temperature"]
            c_e_flux = pybamm.BoundaryGradient(c_e, "right")
            flux_left = -i_boundary_cc * pybamm.BoundaryValue(1 / sigma_eff, "left")
            flux_right = (
                (i_boundary_cc / pybamm.BoundaryValue(conductivity, "right"))
                - pybamm.BoundaryValue(
                    (1 + param.Theta * T) * param.chi(c_e) / c_e, "right"
                )
                * c_e_flux
                - i_boundary_cc * pybamm.BoundaryValue(1 / sigma_eff, "right")
            )

            lbc = (flux_left, "Neumann")
            rbc = (flux_right, "Neumann")
            lbc_c_e = (pybamm.Scalar(0), "Neumann")
            rbc_c_e = (c_e_flux, "Neumann")

        elif self.domain == "Positive":
            T = variables["Positive electrode temperature"]
            c_e_flux = pybamm.BoundaryGradient(c_e, "left")
            flux_left = (
                (i_boundary_cc / pybamm.BoundaryValue(conductivity, "left"))
                - pybamm.BoundaryValue(
                    (1 + param.Theta * T) * param.chi(c_e) / c_e, "left"
                )
                * c_e_flux
                - i_boundary_cc * pybamm.BoundaryValue(1 / sigma_eff, "left")
            )
            flux_right = -i_boundary_cc * pybamm.BoundaryValue(1 / sigma_eff, "right")

            lbc = (flux_left, "Neumann")
            rbc = (flux_right, "Neumann")
            lbc_c_e = (c_e_flux, "Neumann")
            rbc_c_e = (pybamm.Scalar(0), "Neumann")

        # TODO: check why we still need the boundary conditions for c_e, once we have
        # internal boundary conditions
        self.boundary_conditions = {
            delta_phi: {"left": lbc, "right": rbc},
            c_e: {"left": lbc_c_e, "right": rbc_c_e},
        }

        if self.domain == "Negative":
            phi_e = variables["Electrolyte potential"]
            self.boundary_conditions.update(
                {
                    phi_e: {
                        "left": (pybamm.Scalar(0), "Neumann"),
                        "right": (pybamm.Scalar(0), "Neumann"),
                    }
                }
            )


class FullAlgebraic(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations. (Full refers to unreduced by
    asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


     **Extends:** :class:`pybamm.electrolyte_conductivity.surface_potential_form.BaseFull`
    """  # noqa: E501

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def set_algebraic(self, variables):
        if self.domain == "Separator":
            return

        delta_phi = variables[self.domain + " electrode surface potential difference"]
        i_e = variables[self.domain + " electrolyte current density"]

        # Variable summing all of the interfacial current densities
        sum_j = variables[
            "Sum of " + self.domain.lower() + " electrode interfacial current densities"
        ]

        self.algebraic[delta_phi] = pybamm.div(i_e) - sum_j


class FullDifferential(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations and where capacitance is present.
    (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.electrolyte_conductivity.surface_potential_form.BaseFull`

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

        # Variable summing all of the interfacial current densities
        sum_j = variables[
            "Sum of " + self.domain.lower() + " electrode interfacial current densities"
        ]

        self.rhs[delta_phi] = 1 / C_dl * (pybamm.div(i_e) - sum_j)
