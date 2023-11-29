#
# Class for lithium plating
#
import pybamm
from .base_plating import BasePlating


class Plating(BasePlating):
    """Class for lithium plating, from :footcite:t:`OKane2020` and
    :footcite:t:`OKane2022`.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    x_average : bool
        Whether to use x-averaged variables (SPM, SPMe, etc) or full variables (DFN)
    options : dict, optional
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, x_average, options):
        super().__init__(param, domain, options=options)
        self.x_average = x_average
        pybamm.citations.register("OKane2020")
        pybamm.citations.register("OKane2022")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain
        if self.x_average is True:
            c_plated_Li_av = pybamm.Variable(
                f"X-averaged {domain} lithium plating concentration [mol.m-3]",
                domain="current collector",
                scale=self.param.c_Li_typ,
            )
            c_plated_Li = pybamm.PrimaryBroadcast(c_plated_Li_av, f"{domain} electrode")
            c_dead_Li_av = pybamm.Variable(
                f"X-averaged {domain} dead lithium concentration [mol.m-3]",
                domain="current collector",
            )
            c_dead_Li = pybamm.PrimaryBroadcast(c_dead_Li_av, f"{domain} electrode")
        else:
            c_plated_Li = pybamm.Variable(
                f"{Domain} lithium plating concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=self.param.c_Li_typ,
            )
            c_dead_Li = pybamm.Variable(
                f"{Domain} dead lithium concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
            )

        variables = self._get_standard_concentration_variables(c_plated_Li, c_dead_Li)

        return variables

    def get_coupled_variables(self, variables):
        param = self.param
        domain, Domain = self.domain_Domain
        delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
        c_e_n = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
        T = variables[f"{Domain} electrode temperature [K]"]
        eta_sei = variables[f"{Domain} electrode SEI film overpotential [V]"]
        c_plated_Li = variables[f"{Domain} lithium plating concentration [mol.m-3]"]
        j0_stripping = param.j0_stripping(c_e_n, c_plated_Li, T)
        j0_plating = param.j0_plating(c_e_n, c_plated_Li, T)

        eta_stripping = delta_phi + eta_sei
        eta_plating = -eta_stripping
        F_RT = param.F / (param.R * T)
        # NEW: transfer coefficients can be set by the user
        alpha_stripping = self.param.alpha_stripping
        alpha_plating = self.param.alpha_plating

        lithium_plating_option = getattr(self.options, domain)["lithium plating"]
        if lithium_plating_option in ["reversible", "partially reversible"]:
            j_stripping = j0_stripping * pybamm.exp(
                F_RT * alpha_stripping * eta_stripping
            ) - j0_plating * pybamm.exp(F_RT * alpha_plating * eta_plating)
        elif lithium_plating_option == "irreversible":
            # j_stripping is always negative, because there is no stripping, only
            # plating
            j_stripping = -j0_plating * pybamm.exp(F_RT * alpha_plating * eta_plating)

        variables.update(self._get_standard_overpotential_variables(eta_stripping))
        variables.update(self._get_standard_reaction_variables(j_stripping))

        # Update whole cell variables, which also updates the "sum of" variables
        variables.update(super().get_coupled_variables(variables))

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        if self.x_average is True:
            c_plated_Li = variables[
                f"X-averaged {domain} lithium plating concentration [mol.m-3]"
            ]
            c_dead_Li = variables[
                f"X-averaged {domain} dead lithium concentration [mol.m-3]"
            ]
            a_j_stripping = variables[
                f"X-averaged {domain} lithium plating volumetric "
                "interfacial current density [A.m-3]"
            ]
            L_sei = variables[f"X-averaged {domain} total SEI thickness [m]"]
        else:
            c_plated_Li = variables[f"{Domain} lithium plating concentration [mol.m-3]"]
            c_dead_Li = variables[f"{Domain} dead lithium concentration [mol.m-3]"]
            a_j_stripping = variables[
                f"{Domain} lithium plating volumetric "
                "interfacial current density [A.m-3]"
            ]
            L_sei = variables[f"{Domain} total SEI thickness [m]"]

        lithium_plating_option = getattr(self.options, domain)["lithium plating"]
        if lithium_plating_option == "reversible":
            # In the reversible plating model, there is no dead lithium
            dc_plated_Li = -a_j_stripping / self.param.F
            dc_dead_Li = pybamm.Scalar(0)
        elif lithium_plating_option == "irreversible":
            # In the irreversible plating model, all plated lithium is dead lithium
            dc_plated_Li = pybamm.Scalar(0)
            dc_dead_Li = -a_j_stripping / self.param.F
        elif lithium_plating_option == "partially reversible":
            # In the partially reversible plating model, the coupling term turns
            # reversible lithium into dead lithium over time.
            dead_lithium_decay_rate = self.param.dead_lithium_decay_rate(L_sei)
            coupling_term = dead_lithium_decay_rate * c_plated_Li
            dc_plated_Li = -a_j_stripping / self.param.F - coupling_term
            dc_dead_Li = coupling_term

        self.rhs = {
            c_plated_Li: dc_plated_Li,
            c_dead_Li: dc_dead_Li,
        }

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        if self.x_average is True:
            c_plated_Li = variables[
                f"X-averaged {domain} lithium plating concentration [mol.m-3]"
            ]
            c_dead_Li = variables[
                f"X-averaged {domain} dead lithium concentration [mol.m-3]"
            ]
        else:
            c_plated_Li = variables[f"{Domain} lithium plating concentration [mol.m-3]"]
            c_dead_Li = variables[f"{Domain} dead lithium concentration [mol.m-3]"]
        c_plated_Li_0 = self.param.c_plated_Li_0
        zero = pybamm.Scalar(0)

        self.initial_conditions = {c_plated_Li: c_plated_Li_0, c_dead_Li: zero}
