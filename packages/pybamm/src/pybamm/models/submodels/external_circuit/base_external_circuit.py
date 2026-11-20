#
# Base model for the external circuit
#
import pybamm


class BaseModel(pybamm.BaseSubModel):
    """Model to represent the behaviour of the external circuit."""

    def __init__(self, param, options):
        super().__init__(param, options=options)

    def _get_current_variables(self, I):
        """
        Variables derived from the current.

        Parameters
        ----------
        I : :class:`pybamm.Symbol`
            The current [A].

        Returns
        -------
        dict
            The current, current density and C-rate.
        """
        return {
            "Total current density [A.m-2]": I / self.param.A_cc,
            "Current [A]": I,
            "C-rate": I / self.param.Q,
        }
