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
        Whether the particle concentration is averaged over the x-direction.
        Default is False.
    """

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

    def _get_hysteresis_ocp_variables(self, hysteresis_variables):
        """
        Open-circuit potential variables for the given hysteresis state variables.

        Parameters
        ----------
        hysteresis_variables : dict
            The hysteresis state variables, as returned by
            :meth:`_get_hysteresis_state_variables`.
        """
        domain, Domain = self.domain_Domain
        domain_options = getattr(self.options, domain)
        phase_name = self.phase_name
        variables = {}

        if self.reaction == "lithium-ion main":
            # Get equilibrium, delithiation and lithiation OCPs
            sto_surf, sto_bulk, T, T_bulk = self._get_stoichiometry_and_temperature()
            U_eq = self.phase_param.U(sto_surf, T)
            U_lith = self.phase_param.U(sto_surf, T, "lithiation")
            U_lith_bulk = self.phase_param.U(sto_bulk, T_bulk, "lithiation")
            U_delith = self.phase_param.U(sto_surf, T, "delithiation")
            U_delith_bulk = self.phase_param.U(sto_bulk, T_bulk, "delithiation")

            H = U_lith - U_delith

            # Under particle-size distribution the OCPs above live on the
            # "<domain> particle size" domain. Publish the distribution-named
            # variables at that shape, and size-average before publishing under
            # the electrode-level names so they line up with other
            # electrode-domain variables (e.g. in the thermal submodel).
            if domain_options["particle size"] == "distribution":
                variables.update(
                    {
                        f"{Domain} electrode {phase_name}OCP hysteresis distribution [V]": H,
                        f"X-averaged {domain} electrode {phase_name}OCP hysteresis distribution [V]": pybamm.x_average(
                            H
                        ),
                        f"{Domain} electrode {phase_name}equilibrium open-circuit potential distribution [V]": U_eq,
                        f"X-averaged {domain} electrode {phase_name}equilibrium open-circuit potential distribution [V]": pybamm.x_average(
                            U_eq
                        ),
                    }
                )
                U_eq = pybamm.size_average(U_eq)
                H = pybamm.size_average(H)

            variables.update(
                {
                    f"{Domain} electrode {phase_name}OCP hysteresis [V]": H,
                    f"X-averaged {domain} electrode {phase_name}OCP hysteresis [V]": pybamm.x_average(
                        H
                    ),
                    f"{Domain} electrode {phase_name}equilibrium open-circuit potential [V]": U_eq,
                    f"X-averaged {domain} electrode {phase_name}equilibrium open-circuit potential [V]": pybamm.x_average(
                        U_eq
                    ),
                }
            )

            # Get correct hysteresis state variable
            state_name = f"electrode {phase_name}hysteresis state"
            if domain_options["particle size"] == "distribution":
                state_name += " distribution"
            h = hysteresis_variables[f"{Domain} {state_name}"]
            h_x_av = hysteresis_variables[f"X-averaged {domain} {state_name}"]

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

    def _get_hysteresis_state(self, variables):
        """The hysteresis state variable, x-averaged if the state is."""
        domain, Domain = self.domain_Domain
        name = f"electrode {self.phase_name}hysteresis state"
        if self.x_average is True:
            return variables[f"X-averaged {domain} {name}"]
        return variables[f"{Domain} {name}"]

    def _get_hysteresis_rate_inputs(self):
        """
        Volumetric interfacial current density, surface stoichiometry and
        temperature driving the hysteresis state, x-averaged if the state is.
        """
        phase_name = self.phase_name
        x_average = self.x_average
        i_vol = self._coupled_electrode_variable(
            f"electrode {phase_name}volumetric interfacial current density [A.m-3]",
            x_average,
        )
        sto_surf = self._coupled_electrode_variable(
            f"{phase_name}particle surface stoichiometry", x_average
        )
        T = self._coupled_electrode_variable("electrode temperature [K]", x_average)
        return i_vol, sto_surf, T

    def _set_hysteresis_initial_conditions(self, h):
        h_init = self.phase_param.h_init
        if self.x_average is True:
            h_init = pybamm.x_average(h_init)
        self.initial_conditions[h] = h_init
