#
# Class for constant porosity
#
import pybamm

from .base_porosity import BaseModel


class Constant(BaseModel):
    """Submodel for constant porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel


    **Extends:** :class:`pybamm.porosity.BaseModel`
    """

    def get_fundamental_variables(self):

        eps_n_av = self.param.epsilon_n
        eps_s_av = self.param.epsilon_s
        eps_p_av = self.param.epsilon_p

        eps_n = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(eps_n_av, ["negative electrode"]),
            "current collector",
        )
        eps_s = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(eps_s_av, ["separator"]), "current collector"
        )
        eps_p = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(eps_p_av, ["positive electrode"]),
            "current collector",
        )
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        zero = pybamm.Scalar(0)
        deps_n_dt = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(zero, ["negative electrode"]), "current collector"
        )
        deps_s_dt = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(zero, ["separator"]), "current collector"
        )
        deps_p_dt = pybamm.SecondaryBroadcast(
            pybamm.PrimaryBroadcast(zero, ["positive electrode"]), "current collector"
        )
        deps_dt = pybamm.Concatenation(deps_n_dt, deps_s_dt, deps_p_dt)

        variables = self._get_standard_porosity_variables(eps)
        variables.update(self._get_standard_porosity_change_variables(deps_dt))

        return variables
