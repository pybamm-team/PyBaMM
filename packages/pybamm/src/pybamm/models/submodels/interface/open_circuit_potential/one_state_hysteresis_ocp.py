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
        Whether the particle concentration, and hence the hysteresis state, is
        averaged over the x-direction. Default is False.
    x_average_particles : bool, optional
        Whether the particle concentrations the open-circuit potential depends on
        are uniform in x. Default is ``x_average``.
    """

    def __init__(
        self,
        param,
        domain,
        reaction,
        options,
        phase="primary",
        x_average=False,
        x_average_particles=None,
    ):
        warnings.warn(
            "The definition of the hysteresis decay rate parameter has changed in "
            "PyBaMM v25.10. Please see the CHANGELOG for more details.",
            stacklevel=2,
        )
        super().__init__(
            param, domain, reaction, options, phase, x_average, x_average_particles
        )
        pybamm.citations.register("Axen2022")

    def build(self):
        variables = self._get_hysteresis_state_variables()
        variables.update(self._get_hysteresis_ocp_variables(variables))
        self.variables.update(variables)

        h = self._get_hysteresis_state(variables)
        i_vol, sto_surf, T = self._get_hysteresis_rate_inputs()
        c_max = self.phase_param.c_max
        eps = self.phase_param.epsilon_s
        if self.x_average is True:
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
        self._set_hysteresis_initial_conditions(h)
