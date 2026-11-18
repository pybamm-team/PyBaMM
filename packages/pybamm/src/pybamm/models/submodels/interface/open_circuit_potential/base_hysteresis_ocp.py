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

        if domain_options["particle size"] == "distribution":
            return self._get_hysteresis_state_distribution_variables()

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
        return {
            f"{Domain} electrode {phase_name}hysteresis state": h,
            f"X-averaged {domain} electrode {phase_name}hysteresis state": h_x_av,
        }

    def _get_hysteresis_state_distribution_variables(self):
        """
        Under a particle-size distribution each particle size carries its own
        hysteresis state, so particles can sit on different branches (e.g. at
        rest, when lithium redistributes between sizes). The electrode-level
        state is the (area-weighted) size average.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        size = f"{domain} {phase_name}particle size"

        if self.x_average is False:
            h_dist = pybamm.Variable(
                f"{Domain} electrode {phase_name}hysteresis state distribution",
                domains={
                    "primary": size,
                    "secondary": f"{domain} electrode",
                    "tertiary": "current collector",
                },
            )
            h_dist_x_av = pybamm.x_average(h_dist)
            h = pybamm.size_average(h_dist)
            h_x_av = pybamm.x_average(h)
        else:
            h_dist_x_av = pybamm.Variable(
                f"X-averaged {domain} electrode {phase_name}"
                "hysteresis state distribution",
                domains={"primary": size, "secondary": "current collector"},
            )
            h_dist = pybamm.SecondaryBroadcast(h_dist_x_av, [f"{domain} electrode"])
            h_x_av = pybamm.size_average(h_dist_x_av)
            h = pybamm.PrimaryBroadcast(h_x_av, [f"{domain} electrode"])
        return {
            f"{Domain} electrode {phase_name}hysteresis state": h,
            f"X-averaged {domain} electrode {phase_name}hysteresis state": h_x_av,
            f"{Domain} electrode {phase_name}hysteresis state distribution": h_dist,
            f"X-averaged {domain} electrode {phase_name}"
            "hysteresis state distribution": h_dist_x_av,
        }

    def _get_hysteresis_rhs_inputs(self, variables):
        """
        Return the hysteresis state, the rate of change of stoichiometry it is
        driven by (``i_vol / eps`` in [A.m-3], or ``3 j / R`` per particle size
        under a particle-size distribution), and the surface stoichiometry and
        temperature at the same shape.
        """
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        prefix = "" if self.x_average is False else "X-averaged "
        dom = Domain if self.x_average is False else domain
        T = variables[f"{prefix}{dom} electrode temperature [K]"]

        if getattr(self.options, domain)["particle size"] == "distribution":
            j = variables[
                f"{prefix}{dom} electrode {phase_name}"
                "interfacial current density distribution [A.m-2]"
            ]
            R = pybamm.SpatialVariable(
                f"R_{domain[0]}", domains=j.domains, coord_sys="cartesian"
            )
            sto_surf = variables[
                f"{prefix}{dom} {phase_name}particle surface stoichiometry distribution"
            ]
            T = pybamm.PrimaryBroadcast(T, [f"{domain} {phase_name}particle size"])
            h = variables[
                f"{prefix}{dom} electrode {phase_name}hysteresis state distribution"
            ]
            rate = 3 * j / R
        else:
            i_vol = variables[
                f"{prefix}{dom} electrode {phase_name}"
                "volumetric interfacial current density [A.m-3]"
            ]
            eps = self.phase_param.epsilon_s
            if self.x_average:
                eps = pybamm.x_average(eps)
            sto_surf = variables[
                f"{prefix}{dom} {phase_name}particle surface stoichiometry"
            ]
            h = variables[f"{prefix}{dom} electrode {phase_name}hysteresis state"]
            rate = i_vol / eps
        return h, rate, sto_surf, T

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
        domain = self.domain
        h, *_ = self._get_hysteresis_rhs_inputs(variables)
        h_init = self.phase_param.h_init
        if self.x_average:
            h_init = pybamm.x_average(h_init)
        if getattr(self.options, domain)["particle size"] == "distribution":
            h_init = pybamm.PrimaryBroadcast(
                h_init, [f"{domain} {self.phase_name}particle size"]
            )

        self.initial_conditions[h] = h_init
