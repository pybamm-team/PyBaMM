#
# Base class for open-circuit potential with hysteresis
#
import pybamm

from . import BaseOpenCircuitPotential


class BaseHysteresisOpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Base class for open-circuit potentials with single-state hysteresis

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. See
        :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    x_average : bool
        Whether the particle concentration is averaged over the x-direction. Default is False.
    """

    def __init__(
        self, param, domain, reaction, options, phase="primary", x_average=False
    ):
        super().__init__(
            param, domain, reaction, options=options, phase=phase, x_average=x_average
        )

    def _get_hysteresis_state_variables(self):
        domain, Domain = self.domain_Domain
        domain_options = getattr(self.options, domain)
        phase_name = self.phase_name

        if self.x_average is False:
            h = pybamm.Variable(
                f"{Domain} electrode {phase_name}hysteresis state",
                domains={
                    "primary": f"{domain} electrode",
                    "secondary": "current collector",
                },
            )
            h_x_av = pybamm.x_average(h)
        else:
            h_x_av = pybamm.Variable(
                f"X-averaged {domain} electrode {phase_name}hysteresis state",
                domains={
                    "primary": "current collector",
                },
            )
            h = pybamm.PrimaryBroadcast(h_x_av, [f"{domain} electrode"])
        variables = {
            f"{Domain} electrode {phase_name}hysteresis state": h,
            f"X-averaged {domain} electrode {phase_name}hysteresis state": h_x_av,
        }
        if domain_options["particle size"] == "distribution":
            h_dist = pybamm.PrimaryBroadcast(h, [f"{domain} {phase_name}particle size"])
            h_dist_x_av = pybamm.x_average(h_dist)
            variables.update(
                {
                    f"{Domain} electrode {phase_name}hysteresis state distribution": h_dist,
                    f"X-averaged {domain} electrode {phase_name}hysteresis state distribution": h_dist_x_av,
                }
            )
        return variables

    def _get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        domain_options = getattr(self.options, domain)
        phase_name = self.phase_name

        if self.reaction == "lithium-ion main":
            # Get equilibrium, delithiation and lithiation OCPs
            sto_surf, sto_bulk, T, T_bulk = self._get_stoichiometry_and_temperature(
                variables
            )
            U_eq = self.phase_param.U(sto_surf, T)
            U_eq_x_av = self.phase_param.U(sto_surf, T)
            U_lith = self.phase_param.U(sto_surf, T, "lithiation")
            U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk)
            U_delith = self.phase_param.U(sto_surf, T, "delithiation")
            U_delith_bulk = self.phase_param.U(sto_bulk, T_bulk, "delithiation")

            H = U_lith - U_delith
            H_x_av = pybamm.x_average(H)

            variables.update(
                {
                    f"{Domain} electrode {phase_name}OCP hysteresis [V]": H,
                    f"X-averaged {domain} electrode {phase_name}OCP hysteresis [V]": H_x_av,
                    f"{Domain} electrode {phase_name}equilibrium open-circuit potential [V]": U_eq,
                    f"X-averaged {domain} electrode {phase_name}equilibrium open-circuit potential [V]": U_eq_x_av,
                }
            )

            # Get correct hysteresis state variable
            if domain_options["particle size"] == "distribution":
                h = variables[
                    f"{Domain} electrode {phase_name}hysteresis state distribution"
                ]
                h_x_av = variables[
                    f"X-averaged {domain} electrode {phase_name}hysteresis state distribution"
                ]
            else:
                h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
                h_x_av = variables[
                    f"X-averaged {domain} electrode {phase_name}hysteresis state"
                ]

            # check MPM
            if (
                domain_options["particle size"] == "distribution"
                and "current collector" in sto_surf.domains["secondary"]
            ):
                ocp_surf = (1 + h_x_av) / 2 * U_delith + (1 - h_x_av) / 2 * U_lith
            else:
                ocp_surf = (1 + h) / 2 * U_delith + (1 - h) / 2 * U_lith

            # Size average
            h_s_av = pybamm.size_average(h_x_av)
            ocp_bulk = (1 + h_s_av) / 2 * U_delith_bulk + (1 - h_s_av) / 2 * U_lith_bulk
            dUdT = self.phase_param.dUdT(sto_surf)

        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

    def set_initial_conditions(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        h_init = self.phase_param.h_init
        if self.x_average is False:
            h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
        else:
            h = variables[f"X-averaged {domain} electrode {phase_name}hysteresis state"]
            h_init = pybamm.x_average(h_init)

        self.initial_conditions[h] = h_init
