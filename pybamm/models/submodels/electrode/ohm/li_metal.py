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
        domain_param = self.domain_param
        pot_scale = self.param.potential_scale
        delta_phi_s_dim = pot_scale * delta_phi_s

        variables = {
            f"{Domain} electrode potential drop": delta_phi_s,
            f"{Domain} electrode potential drop [V]": delta_phi_s_dim,
            f"X-averaged {domain} electrode ohmic losses": delta_phi_s / 2,
            f"X-averaged {domain} electrode ohmic losses [V]": delta_phi_s_dim / 2,
            "Lithium metal interface electrode potential": phi_s,
            "Lithium metal interface electrode potential [V]": pot_scale * phi_s,
            "Lithium metal interface electrolyte potential": phi_e,
            "Lithium metal interface electrolyte potential [V]": domain_param.U_ref
            + pot_scale * phi_e,
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

    **Extends:** :class:`pybamm.electrode.li_metal.LithiumMetalBaseModel`
    """

    def get_fundamental_variables(self):
        ocp_ref = self.domain_param.U_ref
        pot_scale = self.param.potential_scale

        delta_phi = pybamm.Variable(
            "Lithium metal interface surface potential difference",
            domain="current collector",
        )
        variables = {
            "Lithium metal interface surface potential difference": delta_phi,
            "Lithium metal interface surface potential difference [V]": ocp_ref
            + delta_phi * pot_scale,
        }

        return variables

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = variables["Current collector current density"]
        T = variables[f"{Domain} current collector temperature"]
        l = domain_param.l
        delta_phi_s = i_boundary_cc * l / domain_param.sigma(T)

        phi_s_cc = variables[f"{Domain} current collector potential"]
        delta_phi = variables["Lithium metal interface surface potential difference"]

        # Potentials at the anode/separator interface
        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables

    def set_initial_conditions(self, variables):
        delta_phi = variables["Lithium metal interface surface potential difference"]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}

    def set_rhs(self, variables):
        if self.options["surface form"] == "differential":
            j_pl = variables["Lithium metal plating current density"]
            j_sei = variables["SEI interfacial current density"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]

            C_dl = self.domain_param.C_dl

            self.rhs[delta_phi] = 1 / C_dl * (i_cc - sum_j)

    def set_algebraic(self, variables):
        if self.options["surface form"] != "differential":  # also catches "false"
            j_pl = variables["Lithium metal plating current density"]
            j_sei = variables["SEI interfacial current density"]
            sum_j = j_pl + j_sei

            i_cc = variables["Current collector current density"]
            delta_phi = variables[
                "Lithium metal interface surface potential difference"
            ]

            self.algebraic[delta_phi] = i_cc - sum_j


class LithiumMetalExplicit(LithiumMetalBaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.

    **Extends:** :class:`pybamm.electrode.li_metal.LithiumMetalBaseModel`
    """

    def get_coupled_variables(self, variables):
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = variables["Current collector current density"]
        T = variables[f"{Domain} current collector temperature"]
        l = domain_param.l
        delta_phi_s = i_boundary_cc * l / domain_param.sigma(T)

        phi_s_cc = variables[f"{Domain} current collector potential"]
        delta_phi = variables["Lithium metal interface surface potential difference"]

        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        return variables
