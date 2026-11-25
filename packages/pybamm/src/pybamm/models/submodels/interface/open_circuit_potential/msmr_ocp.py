#
# Open-circuit potential from the Multi-Species Multi-Reaction framework
#
import pybamm

from . import BaseOpenCircuitPotential


class MSMROpenCircuitPotential(BaseOpenCircuitPotential):
    """
    Class for open-circuit potential within the Multi-Species Multi-Reaction
    framework :footcite:t:`Baker2018`. The thermodynamic model is presented in
    :footcite:t:`Verbrugge2017`, along with parameter values for a number of
    substitutional materials.
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
        super().__init__(
            param, domain, reaction, options, phase, x_average, x_average_particles
        )
        pybamm.citations.register("Baker2018")

    def build(self):
        domain = self.domain
        phase_name = self.phase_name
        x_average = self._x_average_inputs

        # MSMR open-circuit potentials only exist for the main reaction
        if self.reaction != "lithium-ion main":
            raise pybamm.OptionError(
                f"MSMR open-circuit potential is not defined for {self.reaction!r}"
            )
        T = self._coupled_electrode_variable("electrode temperature [K]", x_average)
        # For "particle-size distribution" models, take distribution version
        # of the surface stoichiometry and potential.
        if self.size_distribution:
            sto_surf = self._coupled_electrode_variable(
                f"{phase_name}particle surface stoichiometry distribution",
                x_average,
                size=True,
            )
            ocp_surf = self._coupled_electrode_variable(
                f"{phase_name}particle surface potential distribution [V]",
                x_average,
                size=True,
            )
            T = pybamm.PrimaryBroadcast(T, [f"{domain} particle size"])
        else:
            sto_surf = self._coupled_electrode_variable(
                f"{phase_name}particle surface stoichiometry", x_average
            )
            ocp_surf = self._coupled_electrode_variable(
                f"{phase_name}particle surface potential [V]", x_average
            )

        ocp_bulk = self.coupled_variable(
            f"Average {domain} {phase_name}particle potential [V]",
            domain="current collector",
        )
        dUdT = self.phase_param.dUdT(sto_surf)

        variables = self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT)
        self._alias_ocp_as_equilibrium(variables)
        self.variables.update(variables)
