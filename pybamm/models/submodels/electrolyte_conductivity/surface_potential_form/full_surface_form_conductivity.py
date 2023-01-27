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
    options : dict, optional
        A dictionary of options to be passed to the model.


    **Extends:** :class:`pybamm.electrolyte_conductivity.BaseElectrolyteConductivity`
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def get_fundamental_variables(self):
        if self.domain == "negative":
            delta_phi = pybamm.standard_variables.delta_phi_n
        elif self.domain == "separator":
            return {}
        elif self.domain == "positive":
            delta_phi = pybamm.standard_variables.delta_phi_p

        variables = self._get_standard_average_surface_potential_difference_variables(
            pybamm.x_average(delta_phi)
        )
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )

        return variables

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        param = self.param

        if self.domain in ["negative", "positive"]:
            conductivity, sigma_eff = self._get_conductivities(variables)
            i_boundary_cc = variables["Current collector current density"]
            c_e = variables[f"{Domain} electrolyte concentration"]
            delta_phi = variables[f"{Domain} electrode surface potential difference"]
            T = variables[f"{Domain} electrode temperature"]

            i_e = conductivity * (
                param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e)
                + pybamm.grad(delta_phi)
                + i_boundary_cc / sigma_eff
            )
            variables[f"{Domain} electrolyte current density"] = i_e
            variables[
                f"Divergence of {self.domain} electrolyte current density"
            ] = pybamm.div(i_e)

            phi_s = variables[f"{Domain} electrode potential"]
            phi_e = phi_s - delta_phi

        elif self.domain == "separator":
            x_s = pybamm.standard_spatial_vars.x_s

            i_boundary_cc = variables["Current collector current density"]
            c_e_s = variables["Separator electrolyte concentration"]
            if self.options.electrode_types["negative"] == "planar":
                phi_e_n_s = variables["Lithium metal interface electrolyte potential"]
            else:
                phi_e_n = variables["Negative electrolyte potential"]
                phi_e_n_s = pybamm.boundary_value(phi_e_n, "right")
            tor_s = variables["Separator porosity"]
            T = variables["Separator temperature"]

            chiRT_over_Fc_e_s = param.chiRT_over_Fc(c_e_s, T)
            kappa_s_eff = param.kappa_e(c_e_s, T) * tor_s

            phi_e = phi_e_n_s + pybamm.IndefiniteIntegral(
                chiRT_over_Fc_e_s * pybamm.grad(c_e_s)
                - param.C_e * i_boundary_cc / kappa_s_eff,
                x_s,
            )

            i_e = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")
            variables[f"{Domain} electrolyte current density"] = i_e

            # Update boundary conditions (for indefinite integral)
            self.boundary_conditions[c_e_s] = {
                "left": (pybamm.boundary_gradient(c_e_s, "left"), "Neumann"),
                "right": (pybamm.boundary_gradient(c_e_s, "right"), "Neumann"),
            }

        variables[f"{Domain} electrolyte potential"] = phi_e

        if self.domain == "positive":
            phi_e_dict = {}
            i_e_dict = {}
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize().split()[0]
                phi_e_dict[domain] = variables[f"{Domain} electrolyte potential"]
                i_e_dict[domain] = variables[f"{Domain} electrolyte current density"]

            variables.update(self._get_standard_potential_variables(phi_e_dict))

            i_e = pybamm.concatenation(*i_e_dict.values())
            variables.update(self._get_standard_current_variables(i_e))
            variables.update(self._get_electrolyte_overpotentials(variables))

        # save boundary conditons as variables
        if self.domain == "negative":
            grad_c_e = pybamm.boundary_gradient(c_e, "right")
            grad_left = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            grad_right = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "right"))
                - pybamm.boundary_value(param.chiRT_over_Fc(c_e, T), "right") * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")
            )

        elif self.domain == "positive":
            T = variables["Positive electrode temperature"]
            grad_c_e = pybamm.boundary_gradient(c_e, "left")
            grad_left = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "left"))
                - pybamm.boundary_value(param.chiRT_over_Fc(c_e, T), "left") * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            )
            grad_right = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")

        if self.domain in ["negative", "positive"]:
            variables.update(
                {
                    f"{self.domain} grad(delta_phi) left": grad_left,
                    f"{self.domain} grad(delta_phi) right": grad_right,
                    f"{self.domain} grad(c_e) internal": grad_c_e,
                }
            )
        return variables

    def _get_conductivities(self, variables):
        Domain = self.domain.capitalize()

        param = self.param
        tor_e = variables[f"{Domain} electrolyte transport efficiency"]
        tor_s = variables[f"{Domain} electrode transport efficiency"]
        c_e = variables[f"{Domain} electrolyte concentration"]
        T = variables[f"{Domain} electrode temperature"]
        sigma = self.domain_param.sigma(T)

        kappa_eff = param.kappa_e(c_e, T) * tor_e
        sigma_eff = sigma * tor_s
        conductivity = kappa_eff / (param.C_e / param.gamma_e + kappa_eff / sigma_eff)

        return conductivity, sigma_eff

    def set_initial_conditions(self, variables):
        Domain = self.domain.capitalize()

        if self.domain == "separator":
            return

        delta_phi_e = variables[f"{Domain} electrode surface potential difference"]
        delta_phi_e_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi_e: delta_phi_e_init}

    def set_boundary_conditions(self, variables):
        Domain = self.domain.capitalize()

        if self.domain == "separator":
            return

        c_e = variables[f"{Domain} electrolyte concentration"]
        delta_phi = variables[f"{Domain} electrode surface potential difference"]

        grad_left = variables[f"{self.domain} grad(delta_phi) left"]
        grad_right = variables[f"{self.domain} grad(delta_phi) right"]
        grad_c_e = variables[f"{self.domain} grad(c_e) internal"]

        lbc = (grad_left, "Neumann")
        rbc = (grad_right, "Neumann")
        if self.domain == "negative":
            lbc_c_e = (pybamm.Scalar(0), "Neumann")
            rbc_c_e = (grad_c_e, "Neumann")
        elif self.domain == "positive":
            lbc_c_e = (grad_c_e, "Neumann")
            rbc_c_e = (pybamm.Scalar(0), "Neumann")

        # TODO: check why we still need the boundary conditions for c_e, once we have
        # internal boundary conditions
        self.boundary_conditions = {
            delta_phi: {"left": lbc, "right": rbc},
            c_e: {"left": lbc_c_e, "right": rbc_c_e},
        }

        if self.domain == "negative":
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
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_conductivity.surface_potential_form.BaseFull`
    """  # noqa: E501

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_algebraic(self, variables):
        domain, Domain = self.domain_Domain

        if self.domain == "separator":
            return

        delta_phi = variables[f"{Domain} electrode surface potential difference"]
        i_e = variables[f"{Domain} electrolyte current density"]

        # Variable summing all of the interfacial current densities
        sum_a_j = variables[
            f"Sum of {domain} electrode volumetric " "interfacial current densities"
        ]

        self.algebraic[delta_phi] = pybamm.div(i_e) - sum_a_j


class FullDifferential(BaseModel):
    """Full model for conservation of charge in the electrolyte employing the
    Stefan-Maxwell constitutive equations and where capacitance is present.
    (Full refers to unreduced by asymptotic methods)

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrolyte_conductivity.surface_potential_form.BaseFull`
    """  # noqa: E501

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def set_rhs(self, variables):
        if self.domain == "separator":
            return

        domain, Domain = self.domain_Domain

        C_dl = self.domain_param.C_dl

        delta_phi = variables[f"{Domain} electrode surface potential difference"]
        i_e = variables[f"{Domain} electrolyte current density"]

        # Variable summing all of the interfacial current densities
        sum_a_j = variables[
            f"Sum of {domain} electrode volumetric interfacial current densities"
        ]
        a = variables[f"{Domain} electrode surface area to volume ratio"]

        self.rhs[delta_phi] = 1 / (a * C_dl) * (pybamm.div(i_e) - sum_a_j)
