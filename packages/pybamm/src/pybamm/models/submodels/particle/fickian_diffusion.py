#
# Class for particles with Fickian diffusion
#
import pybamm

from .base_particle import BaseParticle


class FickianDiffusion(BaseParticle):
    """
    Class for molar conservation in particles, employing Fick's law

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain of the model either 'Negative' or 'Positive'
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    x_average : bool
        Whether the particle concentration is averaged over the x-direction
    """

    def __init__(self, param, domain, options, phase="primary", x_average=False):
        super().__init__(param, domain, options, phase, x_average)

    def build(self):
        domain, Domain = self.domain_Domain
        phase_name = self.phase_name
        phase_param = self.phase_param
        particle = f"{domain} {phase_name}particle"

        T = self._broadcast_to_particle(self._coupled_temperature())
        j = self._coupled_interfacial_current_density()
        current = self.coupled_variable("Total current density [A.m-2]")

        if self.size_distribution:
            c_s = pybamm.Variable(
                f"{self._x_prefix} {phase_name}particle "
                "concentration distribution [mol.m-3]",
                domains=self._particle_domains,
                bounds=(0, phase_param.c_max),
                scale=phase_param.c_max,
            )
            R, f_v_dist, variables = self._get_size_distribution()
            variables.update(
                self._get_standard_concentration_distribution_variables(c_s)
            )
            # Size-average with the volume-weighted distribution, since the
            # concentration is volume-based (e.g. for the total lithium)
            c_s_electrode = pybamm.Integral(f_v_dist * c_s, R)
            R_nondim = variables[f"{Domain} {phase_name}particle sizes"]
            R_broad_nondim = pybamm.PrimaryBroadcast(R_nondim, [particle])
        else:
            c_s = pybamm.Variable(
                f"{self._x_prefix} {phase_name}particle concentration [mol.m-3]",
                domains=self._particle_domains,
                bounds=(0, phase_param.c_max),
                scale=phase_param.c_max,
            )
            c_s.print_name = f"c_s_{domain[0]}" + ("_xav" if self.x_average else "")
            c_s_electrode = c_s
            variables = {}
            if self.x_average:
                R_nondim = 1
            else:
                R_nondim = self.coupled_variable(
                    f"{Domain} {phase_name}particle radius", domains=self._domains()
                )
            R_broad_nondim = R_nondim
        # The standard variables need the concentration at every point in the
        # electrode
        if self.x_average:
            c_s_electrode = pybamm.SecondaryBroadcast(
                c_s_electrode, [f"{domain} electrode"]
            )
        variables.update(self._get_standard_concentration_variables(c_s_electrode))

        D_eff = self._get_effective_diffusivity(c_s, T, current)
        N_s = -D_eff * pybamm.grad(c_s)
        rhs = -(1 / (R_broad_nondim**2)) * pybamm.div(N_s)
        rbc = -j * R_nondim / self.param.F / pybamm.surf(D_eff)
        variables.update(
            {
                f"{Domain} {phase_name}particle rhs [mol.m-3.s-1]": rhs,
                f"{Domain} {phase_name}particle bc [mol.m-4]": rbc,
            }
        )
        variables.update(self._get_flux_and_diffusivity_variables(N_s, D_eff))
        self.variables.update(variables)

        self.rhs = {c_s: rhs}
        self.boundary_conditions = {
            c_s: {"left": (pybamm.Scalar(0), "Neumann"), "right": (rbc, "Neumann")}
        }
        c_init = phase_param.c_init
        if self.x_average:
            c_init = pybamm.x_average(c_init)
        if self.size_distribution:
            c_init = pybamm.SecondaryBroadcast(c_init, f"{particle} size")
        self.initial_conditions = {c_s: c_init}
