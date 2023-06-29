#
# Subodels for a lithium metal electrode
#
import pybamm
from .base_ohm import BaseModel


class LithiumMetalBaseModel(BaseModel):
    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def _get_li_metal_interface_variables(self, delta_phi_s, phi_s, phi_e):
        domain, Domain = self.domain_Domain
        variables = {
            f"{Domain} electrode potential drop [V]": delta_phi_s,
            f"X-averaged {domain} electrode ohmic losses [V]": delta_phi_s / 2,
            "Lithium metal interface electrode potential [V]": phi_s,
            "Lithium metal interface electrolyte potential [V]": phi_e,
        }
        return variables


class LithiumMetalSurfaceForm(LithiumMetalBaseModel):
    """Model for potential drop across a lithium metal electrode, with a
    differential or algebraic equation for the surface potential difference

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the submodel, can be "negative" or "positive"
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def get_fundamental_variables(self):
        delta_phi = pybamm.Variable(
            "Lithium metal interface surface potential difference [V]",
            domain="current collector",
        )
        variables = {
            "Lithium metal interface surface potential difference [V]": delta_phi,
        }
        return variables

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        T = variables[f"{Domain} current collector temperature [K]"]
        L = domain_param.L
        delta_phi_s = i_boundary_cc * L / domain_param.sigma(T)

        phi_s_cc = variables[f"{Domain} current collector potential [V]"]
        delta_phi = variables[
            "Lithium metal interface surface potential difference [V]"
        ]

        # Potentials at the anode/separator interface
        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables

    def set_initial_conditions(self, variables):
        delta_phi = variables[
            "Lithium metal interface surface potential difference [V]"
        ]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_rhs(self, variables):
        if self.options["surface form"] == "differential":
            j_pl = variables["Lithium metal plating current density [A.m-2]"]
            j_sei = variables["SEI interfacial current density [A.m-2]"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density [A.m-2]"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]

            C_dl = self.domain_param.C_dl

            self.rhs[delta_phi] = 1 / C_dl * (i_cc - sum_j)

    def set_algebraic(self, variables):
        if self.options["surface form"] != "differential":  # also catches "false"
            j_pl = variables["Lithium metal plating current density [A.m-2]"]
            j_sei = variables["SEI interfacial current density [A.m-2]"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density [A.m-2]"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]

            self.algebraic[delta_phi] = (i_cc - sum_j) / self.param.I_typ


class LithiumMetalExplicit(LithiumMetalBaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        T = variables[f"{Domain} current collector temperature [K]"]
        L = domain_param.L
        delta_phi_s = i_boundary_cc * L / domain_param.sigma(T)

        phi_s_cc = variables[f"{Domain} current collector potential [V]"]
        delta_phi = variables[
            "Lithium metal interface surface potential difference [V]"
        ]

        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables
