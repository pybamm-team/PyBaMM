import warnings

import pybamm

from . import BaseHysteresisOpenCircuitPotential

# Only throw the hysteresis decay rate warning once
warnings.filterwarnings(
    "once",
    message="The definition of the hysteresis decay rate parameter has changed*",
)


class OneStateHysteresisOpenCircuitPotential(BaseHysteresisOpenCircuitPotential):
    """
    Class for one-state hysteresis model based on the implementation in :footcite:t:`Axen2022`.
    The hysteresis state variable `h` is governed by an ODE which depends on the local surface
    stoichiometry and volumetric interfacial current density.

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
        warnings.warn(
            "The definition of the hysteresis decay rate parameter has changed in "
            "PyBaMM v25.10. Please see the CHANGELOG for more details.",
            stacklevel=2,
        )
        super().__init__(
            param, domain, reaction, options=options, phase=phase, x_average=x_average
        )
        pybamm.citations.register("Axen2022")

    def get_fundamental_variables(self):
        return self._get_hysteresis_state_variables()

    def get_coupled_variables(self, variables):
        return self._get_coupled_variables(variables)

    def set_rhs(self, variables):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name

        c_max = self.phase_param.c_max
        eps = self.phase_param.epsilon_s
        if self.x_average is False:
            i_vol = variables[
                f"{Domain} electrode {phase_name}volumetric interfacial current density [A.m-3]"
            ]
            sto_surf = variables[f"{Domain} {phase_name}particle surface stoichiometry"]
            T = variables[f"{Domain} electrode temperature [K]"]
            h = variables[f"{Domain} electrode {phase_name}hysteresis state"]
        else:
            i_vol = variables[
                f"X-averaged {domain} electrode {phase_name}volumetric interfacial current density [A.m-3]"
            ]
            sto_surf = variables[
                f"X-averaged {domain} {phase_name}particle surface stoichiometry"
            ]
            T = variables[f"X-averaged {domain} electrode temperature [K]"]
            h = variables[f"X-averaged {domain} electrode {phase_name}hysteresis state"]
            eps = pybamm.x_average(eps)
        gamma_lith = self.phase_param.hysteresis_decay(sto_surf, T, "lithiation")
        gamma_delith = self.phase_param.hysteresis_decay(sto_surf, T, "delithiation")

        i_vol_sign = pybamm.sign(i_vol)
        signed_h = (1 - i_vol_sign * h) / 2
        gamma = (
            0.5 * (1 - i_vol_sign) * gamma_lith + 0.5 * (1 + i_vol_sign) * gamma_delith
        )

        dhdt = gamma * i_vol / (self.param.F * c_max * eps) * signed_h

        self.rhs[h] = dhdt
