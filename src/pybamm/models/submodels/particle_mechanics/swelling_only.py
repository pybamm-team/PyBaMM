#
# Class for swelling only (no cracking)
#
import pybamm

from .base_mechanics import BaseMechanics


class SwellingOnly(BaseMechanics):
    """
    Class for swelling only (no cracking), from :footcite:t:`Ai2019`.

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

    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, options, phase)

        pybamm.citations.register("Ai2019")
        pybamm.citations.register("Deshpande2012")

    def get_fundamental_variables(self):
        domain, Domain = self.domain_Domain

        zero = pybamm.FullBroadcast(
            pybamm.Scalar(0), f"{domain} electrode", "current collector"
        )
        zero_av = pybamm.x_average(zero)
        variables = self._get_standard_variables(zero)
        variables.update(
            {
                f"{Domain} particle cracking rate [m.s-1]": zero,
                f"X-averaged {domain} particle cracking rate [m.s-1]": zero_av,
            }
        )
        return variables

    def get_coupled_variables(self, variables):
        variables.update(self._get_standard_surface_variables(variables))
        variables.update(self._get_mechanical_results(variables))
        if self.size_distribution:
            variables.update(self._get_mechanical_size_distribution_results(variables))
        return variables
