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
        return (2 * (self.param.R * T) / self.param.F / ne) * pybamm.arcsinh(
            j / (2 * j0 * u)
        )
