#
# Inverse linear class
#

from .base_inverse import BaseInverseKinetics


class InverseLinear(BaseInverseKinetics):
    """
    Submodel which implements the inverted form of the linear relation to
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
        return (2 * (self.param.R * T) / self.param.F / ne) * j / (2 * j0 * u)
