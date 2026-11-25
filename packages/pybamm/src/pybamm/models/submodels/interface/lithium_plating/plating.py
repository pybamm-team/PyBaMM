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

    def __init__(self, param, domain, x_average, options, phase="primary"):
        super().__init__(
            param, domain, options=options, phase=phase, x_average=x_average
        )
        pybamm.citations.register("OKane2020")
        pybamm.citations.register("OKane2022")

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        scale = phase_param.c_Li_typ
        if self.x_average is True:
            c_plated_Li_av = pybamm.Variable(
                f"X-averaged {domain} {phase_name}lithium plating concentration "
                "[mol.m-3]",
                domain="current collector",
                scale=scale,
            )
            c_plated_Li = pybamm.PrimaryBroadcast(c_plated_Li_av, f"{domain} electrode")
            c_dead_Li_av = pybamm.Variable(
                f"X-averaged {domain} {phase_name}dead lithium concentration [mol.m-3]",
                domain="current collector",
                scale=scale,
            )
            c_dead_Li = pybamm.PrimaryBroadcast(c_dead_Li_av, f"{domain} electrode")
            c_plated_Li_state, c_dead_Li_state = c_plated_Li_av, c_dead_Li_av
        else:
            c_plated_Li = pybamm.Variable(
                f"{Domain} {phase_name}lithium plating concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=scale,
            )
            c_dead_Li = pybamm.Variable(
                f"{Domain} {phase_name}dead lithium concentration [mol.m-3]",
                domain=f"{domain} electrode",
                auxiliary_domains={"secondary": "current collector"},
                scale=scale,
            )
            c_plated_Li_state, c_dead_Li_state = c_plated_Li, c_dead_Li

        variables = self._get_standard_concentration_variables(c_plated_Li, c_dead_Li)

        delta_phi = self._coupled_electrode_variable(
            "electrode surface potential difference [V]"
        )
        c_e_n = self._coupled_electrode_variable("electrolyte concentration [mol.m-3]")
        T = self._coupled_electrode_variable("electrode temperature [K]")
        eta_sei = self._coupled_electrode_variable(
            f"electrode {phase_name}SEI film overpotential [V]"
        )
        j0_stripping = phase_param.j0_stripping(c_e_n, c_plated_Li, T)
        j0_plating = phase_param.j0_plating(c_e_n, c_plated_Li, T)

        eta_stripping = delta_phi + eta_sei
        eta_plating = -eta_stripping
        F_RT = self.param.F / (self.param.R * T)
        # NEW: transfer coefficients can be set by the user
        alpha_stripping = phase_param.alpha_stripping
        alpha_plating = phase_param.alpha_plating

        lithium_plating_option = getattr(getattr(self.options, domain), self.phase)[
            "lithium plating"
        ]
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
        variables.update(self._get_standard_volumetric_plating_variables(variables))
        self.variables.update(variables)

        # Plated and dead lithium concentrations, x-averaged if the states are
        a_j_stripping = variables[
            f"{'X-averaged ' + domain if self.x_average else Domain} electrode "
            f"{phase_name}lithium plating volumetric interfacial current density "
            "[A.m-3]"
        ]
        if lithium_plating_option == "reversible":
            # In the reversible plating model, there is no dead lithium
            dc_plated_Li = -a_j_stripping / self.param.F
            dc_dead_Li = pybamm.Scalar(0)
        elif lithium_plating_option == "irreversible":
            # In the irreversible plating model, all plated lithium is dead lithium
            dc_plated_Li = pybamm.Scalar(0)
            dc_dead_Li = -a_j_stripping / self.param.F
        else:  # "partially reversible"
            # In the partially reversible plating model, the coupling term turns
            # reversible lithium into dead lithium over time.
            L_sei = self._get_coupled_sei_thickness(x_average=self.x_average)
            dead_lithium_decay_rate = self.phase_param.dead_lithium_decay_rate(L_sei)
            coupling_term = dead_lithium_decay_rate * c_plated_Li_state
            dc_plated_Li = -a_j_stripping / self.param.F - coupling_term
            dc_dead_Li = coupling_term

        self.rhs = {
            c_plated_Li_state: dc_plated_Li,
            c_dead_Li_state: dc_dead_Li,
        }

        c_plated_Li_0 = self.phase_param.c_plated_Li_0
        zero = 0 * c_plated_Li_0
        self.initial_conditions = {
            c_plated_Li_state: c_plated_Li_0,
            c_dead_Li_state: zero,
        }
