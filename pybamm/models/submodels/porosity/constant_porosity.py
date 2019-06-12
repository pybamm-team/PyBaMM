#
# Class for constant porosity
#
import pybamm


class ConstantPorosity(pybamm.BaseSubModel):
    """Base class for constant porosity

    Parameters
    ----------
    param : parameter class
        The parameters to use for this submodel

    *Extends:* :class:`pybamm.BaseSubModel`
    """

    def get_fundamental_variables(self):

        eps_n_av = pybamm.param.epsilon_n
        eps_s_av = pybamm.param.epsilon_s
        eps_p_av = pybamm.param.epsilon_p

        eps_n = pybamm.Broadcast(eps_n_av, ["negative electrode"])
        eps_s = pybamm.Broadcast(eps_s_av, ["separator"])
        eps_p = pybamm.Broadcast(eps_p_av, ["positive electrode"])
        eps = pybamm.Concatenation(eps_n, eps_s, eps_p)

        deps_n_dt = pybamm.Scalar(0, ["negative electrode"])
        deps_s_dt = pybamm.Scalar(0, ["separator"])
        deps_p_dt = pybamm.Scalar(0, ["positive electrode"])
        deps_dt = pybamm.Concatenation(deps_n_dt, deps_s_dt, deps_p_dt)

        variables = self._get_standard_porosity_variables(eps)
        variables.update(self._get_standard_porosity_change_variables(deps_dt))

        return variables
