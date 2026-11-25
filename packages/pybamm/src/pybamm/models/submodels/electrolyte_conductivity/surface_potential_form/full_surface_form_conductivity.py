#
# Class for full surface form electrolyte conductivity employing stefan-maxwell
#
import pybamm
from pybamm.models.submodels.electrolyte_conductivity.base_electrolyte_conductivity import (
    BaseElectrolyteConductivity,
)


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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def build(self):
        if self.domain == "separator":
            self._build_separator()
        else:
            self._build_electrode()
        # The positive electrode submodel also sets the whole-cell variables
        if self.domain == "positive":
            self._build_whole_cell()

    def _build_electrode(self):
        domain, Domain = self.domain_Domain
        electrode = f"{domain} electrode"
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        c_e = self._coupled_domain_variable(
            f"{Domain} electrolyte concentration [mol.m-3]", electrode
        )
        T = self._coupled_domain_variable(
            f"{Domain} electrode temperature [K]", electrode
        )
        phi_s = self._coupled_domain_variable(
            f"{Domain} electrode potential [V]", electrode
        )

        delta_phi = pybamm.Variable(
            f"{Domain} electrode surface potential difference [V]",
            domain=electrode,
            auxiliary_domains={"secondary": "current collector"},
            reference=self.domain_param.prim.U_init,
        )
        conductivity, sigma_eff = self._get_conductivities(c_e, T)
        chiRT_over_Fc = self.param.chiRT_over_Fc(c_e, T)
        i_e = conductivity * (
            chiRT_over_Fc * pybamm.grad(c_e)
            + pybamm.grad(delta_phi)
            + i_boundary_cc / sigma_eff
        )
        phi_e = phi_s - delta_phi

        # Neumann conditions on delta_phi: no electrolyte current at the current
        # collector and no electrode current at the separator
        if domain == "negative":
            grad_c_e = pybamm.boundary_gradient(c_e, "right")
            grad_left = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            grad_right = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "right"))
                - pybamm.boundary_value(chiRT_over_Fc, "right") * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")
            )
            c_e_bcs = {
                "left": (pybamm.Scalar(0), "Neumann"),
                "right": (grad_c_e, "Neumann"),
            }
        else:
            grad_c_e = pybamm.boundary_gradient(c_e, "left")
            grad_left = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "left"))
                - pybamm.boundary_value(chiRT_over_Fc, "left") * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            )
            grad_right = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")
            c_e_bcs = {
                "left": (grad_c_e, "Neumann"),
                "right": (pybamm.Scalar(0), "Neumann"),
            }

        variables = self._get_standard_average_surface_potential_difference_variables(
            pybamm.x_average(delta_phi)
        )
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        variables[f"{Domain} electrolyte current density [A.m-2]"] = i_e
        variables.update(
            self._get_standard_domain_potential_variables(electrode, phi_e)
        )
        variables.update(
            {
                f"{domain} grad(delta_phi) left": grad_left,
                f"{domain} grad(delta_phi) right": grad_right,
                f"{domain} grad(c_e) internal": grad_c_e,
            }
        )
        self.variables.update(variables)

        self.initial_conditions[delta_phi] = self.domain_param.prim.U_init
        # TODO: check why we still need the boundary conditions for c_e, once we have
        # internal boundary conditions
        self.boundary_conditions.update(
            {
                delta_phi: {
                    "left": (grad_left, "Neumann"),
                    "right": (grad_right, "Neumann"),
                },
                c_e: c_e_bcs,
            }
        )

    def _build_separator(self):
        i_boundary_cc = self.coupled_variable(
            "Current collector current density [A.m-2]", domain="current collector"
        )
        c_e_s = self._coupled_domain_variable(
            "Separator electrolyte concentration [mol.m-3]", "separator"
        )
        tor_s = self._coupled_domain_variable(
            "Separator electrolyte transport efficiency", "separator"
        )
        T = self._coupled_domain_variable("Separator temperature [K]", "separator")
        planar_negative = self.options.electrode_types["negative"] == "planar"
        if planar_negative:
            phi_e_n_s = self._coupled_lithium_metal_electrolyte_potential()
        else:
            phi_e_n = self._coupled_domain_variable(
                "Negative electrolyte potential [V]", "negative electrode"
            )
            phi_e_n_s = pybamm.boundary_value(phi_e_n, "right")

        chiRT_over_Fc_e_s = self.param.chiRT_over_Fc(c_e_s, T)
        kappa_s_eff = self.param.kappa_e(c_e_s, T) * tor_s
        phi_e_s = phi_e_n_s + pybamm.IndefiniteIntegral(
            chiRT_over_Fc_e_s * pybamm.grad(c_e_s) - i_boundary_cc / kappa_s_eff,
            pybamm.standard_spatial_vars.x_s,
        )
        i_e_s = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")

        variables = {"Separator electrolyte current density [A.m-2]": i_e_s}
        variables.update(
            self._get_standard_domain_potential_variables("separator", phi_e_s)
        )
        if planar_negative:
            phi_e_n = self._get_half_cell_negative_potential({"separator": phi_e_s})
            variables.update(
                self._get_standard_domain_potential_variables(
                    "negative electrode", phi_e_n
                )
            )
        self.variables.update(variables)

        # phi_e BCs live here because the separator exists in half-cells too, and
        # grad(phi_e) needs them to discretise on the outer edges
        self.boundary_conditions[c_e_s] = {
            "left": (pybamm.boundary_gradient(c_e_s, "left"), "Neumann"),
            "right": (pybamm.boundary_gradient(c_e_s, "right"), "Neumann"),
        }
        phi_e = self._coupled_domain_variable(
            "Electrolyte potential [V]", self.options.whole_cell_domains
        )
        self.boundary_conditions.update(
            self._get_electrolyte_potential_boundary_conditions(phi_e)
        )

    def _build_whole_cell(self):
        phi_e_dict = {}
        i_e_dict = {}
        for domain in self.options.whole_cell_domains:
            Domain = domain.split()[0].capitalize()
            phi_e_name = f"{Domain} electrolyte potential [V]"
            i_e_name = f"{Domain} electrolyte current density [A.m-2]"
            if domain == "positive electrode":
                phi_e_dict[domain] = self.variables[phi_e_name]
                i_e_dict[domain] = self.variables[i_e_name]
            else:
                phi_e_dict[domain] = self._coupled_domain_variable(phi_e_name, domain)
                i_e_dict[domain] = self._coupled_domain_variable(i_e_name, domain)

        variables = self._get_standard_whole_cell_potential_variables(phi_e_dict)
        variables["Electrolyte current density [A.m-2]"] = pybamm.concatenation(
            *i_e_dict.values()
        )
        variables.update(self._get_electrolyte_overpotentials(phi_e_dict))
        self.variables.update(variables)

    def _get_conductivities(self, c_e, T):
        """
        Return the effective conductivity of the electrolyte and electrode in
        series, and the effective electrode conductivity.
        """
        domain, Domain = self.domain_Domain
        electrode = f"{domain} electrode"
        tor_e = self._coupled_domain_variable(
            f"{Domain} electrolyte transport efficiency", electrode
        )
        tor_s = self._coupled_domain_variable(
            f"{Domain} electrode transport efficiency", electrode
        )
        if isinstance(self.param, pybamm.LeadAcidParameters):
            # Lead-acid electrodes have no particles
            sto = None
        else:
            phases = getattr(self.options, domain)["particle phases"]
            phase_name = "" if phases == "1" else "primary "
            sto = self._coupled_domain_variable(
                f"{Domain} {phase_name}particle surface stoichiometry", electrode
            )
        sigma = self.domain_param.sigma(sto, T)

        kappa_eff = self.param.kappa_e(c_e, T) * tor_e
        sigma_eff = sigma * tor_s
        conductivity = kappa_eff / (1 + kappa_eff / sigma_eff)

        return conductivity, sigma_eff

    def _get_current_imbalance(self):
        """
        Return the surface potential difference and the imbalance between the
        divergence of the electrolyte current and the interfacial currents.
        """
        domain, Domain = self.domain_Domain
        delta_phi = self.variables[
            f"{Domain} electrode surface potential difference [V]"
        ]
        i_e = self.variables[f"{Domain} electrolyte current density [A.m-2]"]
        sum_a_j = self._coupled_domain_variable(
            f"Sum of {domain} electrode volumetric interfacial current densities "
            "[A.m-3]",
            f"{domain} electrode",
        )
        return delta_phi, pybamm.div(i_e) - sum_a_j


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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def build(self):
        super().build()
        if self.domain == "separator":
            return
        delta_phi, current_imbalance = self._get_current_imbalance()
        self.algebraic[delta_phi] = current_imbalance / self.param.a_j_scale


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
    """

    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options)

    def build(self):
        super().build()
        if self.domain == "separator":
            return
        domain, Domain = self.domain_Domain
        electrode = f"{domain} electrode"
        T = self._coupled_domain_variable(
            f"{Domain} electrode temperature [K]", electrode
        )
        a = self._coupled_domain_variable(
            f"{Domain} electrode surface area to volume ratio [m-1]", electrode
        )
        C_dl = self.domain_param.C_dl(T)
        delta_phi, current_imbalance = self._get_current_imbalance()
        self.rhs[delta_phi] = 1 / (a * C_dl) * current_imbalance
