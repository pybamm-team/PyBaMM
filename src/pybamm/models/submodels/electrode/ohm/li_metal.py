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

    def build(self, submodels):
        delta_phi = pybamm.Variable(
            "Lithium metal interface surface potential difference [V]",
            domain="current collector",
        )
        variables = {
            "Lithium metal interface surface potential difference [V]": delta_phi,
        }
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        T = pybamm.CoupledVariable(
            f"{Domain} current collector temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T.name: T})
        L = domain_param.L
        delta_phi_s = i_boundary_cc * L / domain_param.sigma(T)

        phi_s_cc = pybamm.CoupledVariable(
            f"{Domain} current collector potential [V]",
            domain="current collector",
        )
        self.coupled_variables.update({phi_s_cc.name: phi_s_cc})
        delta_phi = variables[
            "Lithium metal interface surface potential difference [V]"
        ]

        # Potentials at the anode/separator interface
        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables.update(
            self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        )
        delta_phi = variables[
            "Lithium metal interface surface potential difference [V]"
        ]
        delta_phi_init = self.domain_param.prim.U_init

        self.initial_conditions = {delta_phi: delta_phi_init}
        if self.options["surface form"] == "differential":
            j_pl = pybamm.CoupledVariable(
                "Lithium metal plating current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({j_pl.name: j_pl})
            j_sei = pybamm.CoupledVariable(
                f"{Domain} electrode SEI interfacial current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({j_sei.name: j_sei})
            sum_j = j_pl + j_sei

            i_cc = pybamm.CoupledVariable(
                "Current collector current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({i_cc.name: i_cc})
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]
            # temperature at the interface of the negative electrode with the separator
            T_neg = pybamm.CoupledVariable(
                "Negative electrode temperature [K]",
                domain="current collector",
            )
            self.coupled_variables.update({T_neg.name: T_neg})
            T = pybamm.boundary_value(T_neg, "right")

            C_dl = self.domain_param.C_dl(T)

            self.rhs[delta_phi] = 1 / C_dl * (i_cc - sum_j)
        else:  # also catches "false"
            j_pl = pybamm.CoupledVariable(
                "Lithium metal plating current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({j_pl.name: j_pl})
            j_sei = pybamm.CoupledVariable(
                f"{Domain} electrode SEI interfacial current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({j_sei.name: j_sei})
            sum_j = j_pl + j_sei

            i_cc = pybamm.CoupledVariable(
                "Current collector current density [A.m-2]",
                domain="current collector",
            )
            self.coupled_variables.update({i_cc.name: i_cc})
            delta_phi = variables[
                "Lithium metal interface surface potential difference [V]"
            ]

            self.algebraic[delta_phi] = (i_cc - sum_j) / self.param.I_typ
        self.variables.update(variables)


class LithiumMetalExplicit(LithiumMetalBaseModel):
    """Explicit model for potential drop across a lithium metal electrode.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def build(self, submodels):
        Domain = self.domain.capitalize()
        domain_param = self.domain_param

        i_boundary_cc = pybamm.CoupledVariable(
            "Current collector current density [A.m-2]",
            domain="current collector",
        )
        self.coupled_variables.update({i_boundary_cc.name: i_boundary_cc})
        T = pybamm.CoupledVariable(
            f"{Domain} current collector temperature [K]",
            domain="current collector",
        )
        self.coupled_variables.update({T.name: T})
        L = domain_param.L
        delta_phi_s = i_boundary_cc * L / domain_param.sigma(T)

        phi_s_cc = pybamm.CoupledVariable(
            f"{Domain} current collector potential [V]",
            domain="current collector",
        )
        self.coupled_variables.update({phi_s_cc.name: phi_s_cc})
        delta_phi = pybamm.CoupledVariable(
            "Lithium metal interface surface potential difference [V]",
            domain="current collector",
        )
        self.coupled_variables.update({delta_phi.name: delta_phi})

        phi_s = phi_s_cc - delta_phi_s
        phi_e = phi_s - delta_phi

        variables = self._get_li_metal_interface_variables(delta_phi_s, phi_s, phi_e)
        self.variables.update(variables)
