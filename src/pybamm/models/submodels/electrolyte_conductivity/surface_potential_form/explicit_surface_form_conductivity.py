#
# Class for explicit surface form potentials
#
import pybamm
from pybamm.models.submodels.electrolyte_conductivity.base_electrolyte_conductivity import (
    BaseElectrolyteConductivity,
)


class Explicit(BaseElectrolyteConductivity):
    """Class for deriving surface potential difference variables from the electrode
    and electrolyte potentials

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel
    domain : str
        The domain in which the model holds
    options : dict
        A dictionary of options to be passed to the model.
    """

    def __init__(self, param, domain, options):
        super().__init__(param, domain, options)

    def build(self, submodels):
        # skip for separator
        domain, Domain = self.domain_Domain
        if self.domain == "separator":
            return

        Domain = self.domain.capitalize()
        phi_s = pybamm.CoupledVariable(
            f"{Domain} electrode potential [V]",
            f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({phi_s.name: phi_s})
        phi_e = pybamm.CoupledVariable(
            f"{Domain} electrolyte potential [V]",
            f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
        )
        self.coupled_variables.update({phi_e.name: phi_e})
        delta_phi = phi_s - phi_e
        if "positive primary interface" in submodels:
            key = f"{domain} primary interface"
        elif "positive interface" in submodels:
            key = f"{domain} interface"
        elif "leading-order positive interface" in submodels:
            key = f"leading-order {domain} interface"
        else:
            key = "fuck you"

        submodel = submodels.get(key)
        if not isinstance(
            submodel,
            pybamm.models.submodels.interface.kinetics.inverse_kinetics.inverse_butler_volmer.InverseButlerVolmer,
        ):
            variables = self._get_standard_surface_potential_difference_variables(
                delta_phi
            )
        else:
            variables = {}

        delta_phi_av = pybamm.x_average(delta_phi)
        if not isinstance(
            submodel,
            pybamm.models.submodels.interface.kinetics.inverse_kinetics.inverse_butler_volmer.InverseButlerVolmer,
        ):
            variables.update(
                self._get_standard_average_surface_potential_difference_variables(
                    delta_phi_av
                )
            )
        self.variables.update(variables)
