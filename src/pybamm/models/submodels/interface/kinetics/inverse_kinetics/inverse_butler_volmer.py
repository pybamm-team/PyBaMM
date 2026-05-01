#
# Inverse Butler-Volmer class
#
import pybamm

from .base_inverse import BaseInverseKinetics


class InverseButlerVolmer(BaseInverseKinetics):
    """
    Submodel which implements the inverted form of the Butler-Volmer relation to
    solve for the reaction overpotential.

    Parameters
    ----------
    param
        Model parameters
    domain : iter of str, optional
        The domain(s) in which to compute the interfacial current.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model. In this case "SEI film
        resistance" is the important option. See :class:`pybamm.BaseBatteryModel`

    """

    def _get_overpotential(self, j, j0, ne, T, u):
        # Use a specialized arcsinh2(a,b) = arcsinh(a/b) to avoid division by zero
        # errors when j0 is close to zero
        return (2 * (self.param.R * T) / self.param.F / ne) * pybamm.arcsinh2(
            j, 2 * j0 * u
        )

    def _get_pe_shell_potential_drop(self, j_tot, variables):
        # Phase-transformed shell layer in positive electrode particle
        domain = self.domain
        if domain != "positive" or (
            self.options["PE degradation"] != "phase transition"
        ):
            return pybamm.Scalar(0)

        phase_name = self.phase_name
        R_shell = self.phase_param.R_shell
        s_nd_av = variables[
            f"X-averaged {domain} {phase_name}particle moving phase boundary location"
        ]
        R_av = variables[f"X-averaged {domain} {phase_name}particle radius [m]"]
        eta_shell = -j_tot * (pybamm.Scalar(1) - s_nd_av) * R_av * R_shell

        variables.update(self._get_standard_pe_shell_overpotential_variables(eta_shell))
        return eta_shell
