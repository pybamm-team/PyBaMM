#
# Base class for Ohm's law submodels
#
import pybamm
from pybamm.models.submodels.electrode.base_electrode import BaseElectrode


class BaseModel(BaseElectrode):
    """A base class for electrode submodels that employ
    Ohm's law.

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        Either 'negative' or 'positive'
    options : dict, optional
        A dictionary of options to be passed to the model."""

    def __init__(self, param, domain, options=None, set_positive_potential=True):
        super().__init__(param, domain, options, set_positive_potential)

    def _get_electrode_stoichiometry(self, x_averaged=False):
        """
        Surface stoichiometry over the electrode, used for stoichiometry-dependent
        electrode conductivity. Returns None for lead-acid, whose conductivity is a
        function of temperature only.
        """
        if isinstance(self.param, pybamm.LeadAcidParameters):
            return None
        domain, Domain = self.domain_Domain
        if getattr(self.options, domain)["particle phases"] == "1":
            phase_name = ""
        else:
            phase_name = "primary "
        if x_averaged:
            return self._coupled_current_collector_variable(
                f"X-averaged {domain} {phase_name}particle surface stoichiometry"
            )
        return self._coupled_electrode_variable(
            f"{Domain} {phase_name}particle surface stoichiometry"
        )

    def _get_effective_conductivity(self, x_averaged=False):
        """
        Electrode conductivity multiplied by the transport efficiency, over the
        electrode or x-averaged.
        """
        domain, Domain = self.domain_Domain
        if x_averaged:
            coupled = self._coupled_current_collector_variable
            prefix = f"X-averaged {domain}"
        else:
            coupled = self._coupled_electrode_variable
            prefix = Domain
        tor = coupled(f"{prefix} electrode transport efficiency")
        T = coupled(f"{prefix} electrode temperature [K]")
        sto = self._get_electrode_stoichiometry(x_averaged)
        return self.domain_param.sigma(sto, T) * tor

    def _get_x_averaged_surface_potential_difference(self):
        """X-averaged surface potential difference, for x-averaged electrode models."""
        domain, Domain = self.domain_Domain
        if self.options["surface form"] != "false":
            return self._coupled_current_collector_variable(
                f"X-averaged {domain} electrode surface potential difference [V]"
            )
        # With inverse kinetics the explicit surface form redefines the surface
        # potential difference from this potential, so rebuild it from the kinetics
        coupled = self._coupled_electrode_variable
        eta_r = coupled(f"{Domain} electrode reaction overpotential [V]")
        ocp = coupled(f"{Domain} electrode open-circuit potential [V]")
        eta_sei = coupled(f"{Domain} electrode SEI film overpotential [V]")
        return pybamm.x_average(eta_r + ocp - eta_sei)

    def _set_porous_boundary_conditions(self, phi_s, sigma_eff):
        """
        Match the negative electrode potential to the negative current collector
        and the positive electrode current to the current collector current.
        """
        if self.domain == "negative":
            phi_s_cn = self._coupled_current_collector_variable(
                "Negative current collector potential [V]"
            )
            lbc = (phi_s_cn, "Dirichlet")
            rbc = (pybamm.Scalar(0), "Neumann")
        else:
            i_boundary_cc = self._coupled_current_collector_variable(
                "Current collector current density [A.m-2]"
            )
            lbc = (pybamm.Scalar(0), "Neumann")
            rbc = (
                i_boundary_cc / pybamm.boundary_value(-sigma_eff, "right"),
                "Neumann",
            )
        self.boundary_conditions[phi_s] = {"left": lbc, "right": rbc}
