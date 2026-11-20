#
# Subodels for a lithium metal electrode
#
import pybamm

from .base_ohm import BaseModel


class LithiumMetalBaseModel(BaseModel):
    def __init__(self, param, domain, options=None):
        super().__init__(param, domain, options=options)

    def _get_li_metal_interface_variables(self, delta_phi):
        """
        Potentials at the interface between the lithium metal electrode and the
        separator, given the surface potential difference ``delta_phi``.
        """
        domain, Domain = self.domain_Domain
        domain_param = self.domain_param
        coupled = self._coupled_current_collector_variable

        i_boundary_cc = coupled("Current collector current density [A.m-2]")
        T = coupled(f"{Domain} current collector temperature [K]")
        phi_s_cc = coupled(f"{Domain} current collector potential [V]")

        delta_phi_s = i_boundary_cc * domain_param.L / domain_param.sigma(None, T)
        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi
        return {
            f"{Domain} electrode potential drop [V]": delta_phi_s,
            f"X-averaged {domain} electrode ohmic losses [V]": delta_phi_s / 2,
            "Lithium metal interface electrode potential [V]": phi_s,
            "Lithium metal interface electrolyte potential [V]": phi_e,
        }


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

    def build(self):
        Domain = self.domain.capitalize()
        coupled = self._coupled_current_collector_variable

        delta_phi = pybamm.Variable(
            "Lithium metal interface surface potential difference [V]",
            domain="current collector",
        )
        self.variables.update(
            {"Lithium metal interface surface potential difference [V]": delta_phi}
        )
        self.variables.update(self._get_li_metal_interface_variables(delta_phi))

        j_pl = coupled("Lithium metal plating current density [A.m-2]")
        j_sei = coupled(f"{Domain} electrode SEI interfacial current density [A.m-2]")
        i_cc = coupled("Current collector current density [A.m-2]")
        sum_j = j_pl + j_sei
        if self.options["surface form"] == "differential":
            # temperature at the interface of the negative electrode with the separator
            T = pybamm.boundary_value(
                self.coupled_variable(
                    "Negative electrode temperature [K]",
                    domains={
                        "primary": "negative electrode",
                        "secondary": "current collector",
                    },
                ),
                "right",
            )
            C_dl = self.domain_param.C_dl(T)
            self.rhs[delta_phi] = 1 / C_dl * (i_cc - sum_j)
        else:  # also catches "false"
            self.algebraic[delta_phi] = (i_cc - sum_j) / self.param.I_typ

        self.initial_conditions[delta_phi] = self.domain_param.prim.U_init


class LithiumMetalExplicit(LithiumMetalBaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def build(self):
        delta_phi = self._coupled_current_collector_variable(
            "Lithium metal interface surface potential difference [V]"
        )
        self.variables.update(self._get_li_metal_interface_variables(delta_phi))
