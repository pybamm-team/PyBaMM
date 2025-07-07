import pybamm

from . import BaseHysteresisOpenCircuitPotential


class OneStateDifferentialCapacityHysteresisOpenCircuitPotential(
    BaseHysteresisOpenCircuitPotential
):
    """
    Class for one-state open-circuit hysteresis model based on the approach outlined by Wycisk :footcite:t:`Wycisk2022`.
    The hysteresis state variable `h` is governed by an ODE which depends on the local surface stoichiometry,
    volumetric interfacial current density, and differential capacity.

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
        pybamm.citations.register("Wycisk2022")

    def get_fundamental_variables(self):
        return self._get_hysteresis_state_variables()

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        variables = self._get_coupled_variables(variables)

        if self.reaction == "lithium-ion main":
            # determine dQ/dU
            sto_surf, T = self._get_stoichiometry_and_temperature(variables)
            sto_surf = pybamm.size_average(sto_surf)
            T_bulk = pybamm.xyz_average(pybamm.size_average(T))
            if phase_name == "":
                Q_mag = variables[f"{Domain} electrode capacity [A.h]"]
            else:
                Q_mag = variables[
                    f"{Domain} electrode {phase_name}phase capacity [A.h]"
                ]

            dU = self.phase_param.U(sto_surf, T_bulk).diff(sto_surf)
            dQdU = Q_mag / dU
            dQdU_xav = pybamm.x_average(dQdU)
            variables.update(
                {
                    f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]": dQdU,
                    f"X-averaged {domain} electrode {phase_name}differential capacity [A.s.V-1]": dQdU_xav,
                }
            )

        return variables

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        if self.x_average is False:
            i_surf = variables[
                f"{Domain} electrode {phase_name}interfacial current density [A.m-2]"
            ]
            sto_surf = variables[f"{Domain} {phase_name}particle surface stoichiometry"]
            T = variables[f"{Domain} electrode temperature [K]"]
            h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
            dQdU = variables[
                f"{Domain} electrode {phase_name}differential capacity [A.s.V-1]"
            ]
        else:
            i_surf = variables[
                f"X-averaged {domain} electrode {phase_name}interfacial current density [A.m-2]"
            ]
            sto_surf = variables[
                f"X-averaged {domain} {phase_name}particle surface stoichiometry"
            ]
            T = variables[f"X-averaged {domain} electrode temperature [K]"]
            h = variables[f"X-averaged {domain} electrode {phase_name}hysteresis state"]
            dQdU = variables[
                f"X-averaged {domain} electrode {phase_name}differential capacity [A.s.V-1]"
            ]
        # check if composite or not
        if phase_name != "":
            Q_cell = variables[f"{Domain} electrode {phase_name}phase capacity [A.h]"]
        else:
            Q_cell = variables[f"{Domain} electrode capacity [A.h]"]

        Gamma = self.phase_param.hysteresis_decay(sto_surf, T)
        x = self.phase_param.hysteresis_switch

        i_surf_sign = pybamm.sign(i_surf)
        signed_h = 1 - i_surf_sign * h
        gamma = Gamma * (1 / dQdU**x)
        dhdt = gamma * i_surf / Q_cell * signed_h

        self.rhs[h] = dhdt
